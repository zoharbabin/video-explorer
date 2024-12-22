# pylint: disable=no-member
import os
import json
import time
import math
import asyncio
from typing import List, Optional, Dict
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import instructor
from litellm import completion
from pydantic import BaseModel, Field
from kaltura_utils import KalturaUtils

# Load environment variables
load_dotenv()

# In-memory cache for analysis results
analysis_cache: Dict[str, dict] = {}
# In-memory progress tracking
analysis_progress: Dict[str, float] = {}

# Create static directory if it doesn't exist
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)

app = FastAPI(
    title="Video Explorer",
    root_path="",
    servers=[
        {"url": "http://localhost:8000", "description": "Local development server"},
        {"url": "http://127.0.0.1:8000", "description": "Local development server (IP)"},
        {"url": "http://0.0.0.0:8000", "description": "All interfaces"}
    ]
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

# Load configuration values
KALTURA_SESSION_DURATION = int(os.getenv('KALTURA_SESSION_DURATION', 86400))  # 24 hours
PAGE_SIZE = int(os.getenv('PAGE_SIZE', 10))
MODEL_TIMEOUT = int(os.getenv('MODEL_TIMEOUT', 60))
MODEL_MAX_TOKENS = int(os.getenv('MODEL_MAX_TOKENS', 4000))
MODEL_CHUNK_SIZE = int(os.getenv('MODEL_CHUNK_SIZE', 24000))  # Doubled chunk size for fewer API calls
MODEL_TEMPERATURE = float(os.getenv('MODEL_TEMPERATURE', 0))

# Initialize Kaltura client
kaltura = KalturaUtils(
    service_url=os.getenv('KALTURA_SERVICE_URL', 'https://cdnapisec.kaltura.com'),
    partner_id=int(os.getenv('KALTURA_PARTNER_ID')),
    admin_secret=os.getenv('KALTURA_SECRET'),
    session_duration=KALTURA_SESSION_DURATION
)

# Initialize instructor client with litellm
llm_client = instructor.from_litellm(completion)

# Configure litellm to use Bedrock
os.environ["AWS_REGION"] = os.getenv('AWS_REGION', 'us-east-1')

# Define Pydantic models for structured outputs
class TimestampEntry(BaseModel):
    start_timestamp: float = Field(description="Timestamp in seconds")
    end_timestamp: float = Field(description="Timestamp in seconds")
    description: str = Field(description="Description of what occurs at this timestamp")
    topic: str = Field(description="Main topic being discussed at this timestamp")
    importance: int = Field(description="Importance level (1-5)", ge=1, le=5)

class VideoAnalysis(BaseModel):
    summary: str = Field(description="A comprehensive summary of the video content, formatted in clear paragraphs")
    insights: List[str] = Field(description="Key insights from the video, ordered by importance")
    topics: List[dict] = Field(description="Main topics discussed in the video with their relative importance scores")
    timestamps: List[TimestampEntry] = Field(description="Important timestamps with topic-based segmentation")

class ChatResponse(BaseModel):
    answer: str = Field(description="Response to the user's question based on video context")

def init_kaltura_session():
    """Initialize Kaltura session"""
    success, pid = kaltura.init_session()
    if not success:
        print("Failed to initialize Kaltura session")
        return False
    print(f"Successfully initialized Kaltura session for partner ID: {pid}")
    return True

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page"""
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.get("/api/videos")
async def search_videos(category_id: Optional[str] = None, query: Optional[str] = None):
    """Search for videos with English captions"""
    try:
        print("Initializing Kaltura session...")
        if not init_kaltura_session():
            return {"error": "Failed to initialize Kaltura session. Please check your credentials."}
        
        print("Session initialized successfully")
        
        # Use the new fetch_videos method that ensures videos have captions
        videos = kaltura.fetch_videos(
            category_ids=category_id,
            free_text=query,
            number_of_videos=PAGE_SIZE
        )
        
        print(f"Found {len(videos)} videos with captions")
        return {"videos": videos}
        
    except Exception as e:
        print(f"Error in search_videos: {str(e)}")
        return {"error": str(e)}


async def process_transcript_segment(
    segment_chunks: List[dict],
    all_chunks: List[dict],
    segment_index: int,
    total_segments: int,
    video_duration: float
) -> List[TimestampEntry]:
    """
    Analyzes a segment of the video's transcript and extracts key topical moments with timestamps,
    each with a short description, topic, and importance score.
    
    Inputs:
        segment_chunks: A list of chunks (lists of caption entries) that make up this segment.
        all_chunks:     The entire transcript (unused here beyond signature compatibility).
        segment_index:  Index of this segment in the overall segmentation.
        total_segments: Total number of segments in the transcript.
        video_duration: Duration of the video in seconds.
    
    Returns:
        A list of TimestampEntry objects representing key moments:
            [
              {
                "start_timestamp": 120.5,
                "end_timestamp": 134.0,
                "description": "Demonstrates neural network training interface",
                "topic": "Neural Network",
                "importance": 5
              },
              ...
            ]
    """

    try:
        print(f"\n[DEBUG] Processing segment {segment_index + 1}/{total_segments}")
        print(f"[DEBUG] Input chunks: {len(segment_chunks)} chunks with {sum(len(chunk) for chunk in segment_chunks)} total entries")
        
        # --------------------------------------------------------------------
        # 1. Determine start/end times of this segment
        # --------------------------------------------------------------------
        segment_start = round(segment_chunks[0][0]['startTime'] / 1000, 2)
        segment_end   = round(segment_chunks[-1][-1]['endTime'] / 1000, 2)

        # If the segment is extremely short, we can process anyway,
        # but optionally warn (instead of skipping).
        segment_length = segment_end - segment_start
        print(f"[DEBUG] Segment timing: {segment_start:.1f}s -> {segment_end:.1f}s (duration: {segment_length:.1f}s)")
        if segment_length < 15:
            print(f"[DEBUG] Warning: Very short segment (only {segment_length:.2f} seconds).")

        # --------------------------------------------------------------------
        # 2. Build a line-by-line transcript with exact timestamps
        #    e.g. "[120.5s - 125.0s] Let me demonstrate the cloud system."
        # --------------------------------------------------------------------
        timestamped_lines = []
        for chunk in segment_chunks:
            for entry in chunk:
                start_time_s = round(entry['startTime'] / 1000, 2)
                end_time_s   = round(entry['endTime']   / 1000, 2)
                line_text    = entry['text']
                timestamped_lines.append(f"[{start_time_s:.1f}s - {end_time_s:.1f}s] {line_text}")

        # Combine into one string for the prompt
        segment_text = "\n".join(timestamped_lines)

        # --------------------------------------------------------------------
        # 3. Prompt the LLM to analyze the segment and provide key moments
        # --------------------------------------------------------------------
        # We keep the instructions short but clear. The model is told:
        # - Use only the provided timestamps
        # - Identify major transitions or demos
        # - Provide 2-3 key moments, with short, specific descriptions
        # - (Optional) mention spacing, but not too rigid
        prompt = f"""
You have a video segment (Part {segment_index + 1} of {total_segments}), from {segment_start:.1f}s to {segment_end:.1f}s.

TRANSCRIPT (with exact timestamps for each line):
{segment_text}

Instructions:
1. Identify key moments in this segment where primary discussed topics begin and ends.
2. Use the start time from the first line that introduces this topic (e.g., '[120.5s - 134.0s] XXX' -> 120.5).
3. Use the end time from the last line that discusses this topic (e.g., '[120.5s - 134.0s] XXX' -> 134.0).
4. Each key moment should have a short, clear description (10–15 words) that includes at least 4 verbatim words from the original content.
5. Return your answer as a JSON array of TimestampEntry objects. Each object:
   - start_timestamp (float) -> e.g., 120.5
   - end_timestamp (float) -> e.g., 250.5
   - description (string) -> 10-15 words, referencing key terms from that line
   - topic (string) -> The main topic or idea introduced
   - importance (1-5) -> 5 for major transitions, 3-4 for significant points, etc.
6. Make sure there are no overlapping timestamps or duplicates.
7. The list of moments returned should cover the entire segment without gaps in time.
8. Each topic should be at least 60 seconds long.

Example Output:
[
  {{
    "start_timestamp": 120.5,
    "end_timestamp": 234.0,
    "description": "Demonstrates neural network training interface with sample code",
    "topic": "Neural Network",
    "importance": 5
  }},
  {{
    "start_timestamp": 234.0,
    "end_timestamp": 370.0,
    "description": "Explains backpropagation algorithm with real-time model updates",
    "topic": "Backpropagation",
    "importance": 4
  }}
]
"""

        # --------------------------------------------------------------------
        # 4. Call the LLM
        # --------------------------------------------------------------------
        print(f"[DEBUG] Calling LLM for segment analysis...")
        segment_timestamps = await asyncio.to_thread(
            llm_client.chat.completions.create,
            model=os.getenv('MODEL_ID', 'anthropic.claude-3-sonnet-20240229-v1:0'),
            response_model=List[TimestampEntry],
            messages=[{"role": "user", "content": prompt}],
            timeout=MODEL_TIMEOUT,
            max_tokens=MODEL_MAX_TOKENS,
            temperature=MODEL_TEMPERATURE
        )
        print(f"[DEBUG] LLM returned {len(segment_timestamps)} timestamps")

        # --------------------------------------------------------------------
        # 5. Validate the LLM's Results
        # --------------------------------------------------------------------
        print(f"[DEBUG] Starting validation of {len(segment_timestamps)} timestamps")
        valid_results = []
        used_timestamps = set()

        for ts in segment_timestamps:
            print(f"\n[DEBUG] Validating timestamp: {ts.start_timestamp:.1f}s - {ts.end_timestamp:.1f}s")
            print(f"[DEBUG] Description: {ts.description}")
            print(f"[DEBUG] Topic: {ts.topic}")
            print(f"[DEBUG] Importance: {ts.importance}")
            
            # Ensure timestamps are within segment range AND video duration
            if ts.start_timestamp < (segment_start - 0.1):
                print(f"[DEBUG] ❌ Start timestamp {ts.start_timestamp:.1f}s before segment start [{segment_start:.1f}s]")
                continue
            if ts.end_timestamp > (segment_end + 0.1):
                print(f"[DEBUG] ❌ End timestamp {ts.end_timestamp:.1f}s after segment end [{segment_end:.1f}s]")
                continue
            if ts.end_timestamp > (video_duration + 0.1):
                print(f"[DEBUG] ❌ End timestamp {ts.end_timestamp:.1f}s exceeds video duration {video_duration:.1f}s")
                continue
            if ts.start_timestamp >= ts.end_timestamp:
                print(f"[DEBUG] ❌ Start timestamp {ts.start_timestamp:.1f}s not before end timestamp {ts.end_timestamp:.1f}s")
                continue

            # If we've already used this exact timestamp pair, skip
            timestamp_pair = (ts.start_timestamp, ts.end_timestamp)
            if timestamp_pair in used_timestamps:
                print(f"[DEBUG] ❌ Duplicate timestamp pair {ts.start_timestamp:.1f}s - {ts.end_timestamp:.1f}s")
                continue

            # Good enough to accept
            print(f"[DEBUG] ✓ Accepted timestamp pair {ts.start_timestamp:.1f}s - {ts.end_timestamp:.1f}s")
            used_timestamps.add(timestamp_pair)
            valid_results.append(ts)

        print(f"\n[DEBUG] Final validation results:")
        print(f"[DEBUG] - Input timestamps: {len(segment_timestamps)}")
        print(f"[DEBUG] - Valid timestamps: {len(valid_results)}")
        print(f"[DEBUG] - Rejected timestamps: {len(segment_timestamps) - len(valid_results)}")
        
        # Return the final, validated timestamps
        return valid_results

    except Exception as e:
        print(f"\n[DEBUG] ❌ ERROR processing segment {segment_index + 1}:")
        print(f"[DEBUG] Error type: {type(e).__name__}")
        print(f"[DEBUG] Error message: {str(e)}")
        print(f"[DEBUG] Segment details:")
        print(f"[DEBUG] - Start time: {segment_start:.1f}s")
        print(f"[DEBUG] - End time: {segment_end:.1f}s")
        print(f"[DEBUG] - Duration: {segment_length:.1f}s")
        print(f"[DEBUG] - Number of chunks: {len(segment_chunks)}")
        print(f"[DEBUG] - Total entries: {sum(len(chunk) for chunk in segment_chunks)}")
        return []

async def generate_timestamps(transcript_chunks: List[dict], video_duration: float) -> List[TimestampEntry]:
    """Generate timestamps by analyzing the transcript in manageable segments"""
    try:
        # Calculate optimal segment size based on transcript length
        total_text_length = sum(len(entry['text']) for chunk in transcript_chunks for entry in chunk)
        # Aim for segments of ~8000 characters to stay well within context limits
        chars_per_segment = 8000
        chunks_per_segment = max(1, int((chars_per_segment * len(transcript_chunks)) / total_text_length))
        
        # Process transcript in segments
        all_timestamps = []
        segment_count = math.ceil(len(transcript_chunks) / chunks_per_segment)
        
        # Create segment processing tasks
        segment_tasks = []
        for segment_index in range(segment_count):
            start_idx = segment_index * chunks_per_segment
            end_idx = min(start_idx + chunks_per_segment, len(transcript_chunks))
            segment_chunks = transcript_chunks[start_idx:end_idx]
            
            task = process_transcript_segment(
                segment_chunks,
                transcript_chunks,  # Pass full transcript for context
                segment_index,
                segment_count,
                video_duration
            )
            segment_tasks.append(task)
        
        # Process segments in parallel
        print(f"Processing {len(segment_tasks)} transcript segments for timestamps...")
        segment_results = await asyncio.gather(*segment_tasks)
        
        # Combine all timestamps
        all_timestamps = [ts for result in segment_results for ts in result]
        
        # Sort and filter timestamps with minimum spacing
        filtered_timestamps = []
        min_gap = 90  # 1.5 minutes minimum gap
        
        for ts in sorted(all_timestamps, key=lambda x: x.start_timestamp):
            # Validate timestamps are within video duration
            if ts.end_timestamp > video_duration:
                continue
                
            # Check spacing
            if not filtered_timestamps:
                filtered_timestamps.append(ts)
            elif (ts.start_timestamp - filtered_timestamps[-1].end_timestamp) >= min_gap:
                filtered_timestamps.append(ts)
        
        # Ensure reasonable coverage
        target_count = int(video_duration / 180)  # Aim for one timestamp every ~3 minutes
        if len(filtered_timestamps) < target_count * 0.7:  # If we have less than 70% of target
            print(f"Warning: Generated only {len(filtered_timestamps)} timestamps for {video_duration/60:.1f} minute video")
        
        return filtered_timestamps

    except Exception as e:
        print(f"Error generating timestamps: {str(e)}")
        return []

def finalize_segments(chunk_results: List[VideoAnalysis], video_duration: float) -> dict:
    """
    Combine chunk results into a final analysis
    """
    combined_summaries = []
    all_insights = []
    all_topics = []
    
    for chunk in chunk_results:
        combined_summaries.append(chunk.summary)
        all_insights.extend(chunk.insights)
        all_topics.extend(chunk.topics)
    
    # Deduplicate insights and topics
    unique_insights = list(dict.fromkeys(all_insights))
    
    # Combine topic scores and normalize
    topic_scores = {}
    for topic in all_topics:
        name = topic["name"]
        score = topic["importance"]
        topic_scores[name] = max(topic_scores.get(name, 0), score)
    
    final_topics = [
        {"name": name, "importance": score}
        for name, score in sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    
    return {
        "summary": "\n\n".join(combined_summaries),
        "insights": unique_insights,
        "topics": final_topics,
        "timestamps": []  # Will be populated separately
    }

@app.get("/api/analysis-progress/{task_id}")
async def get_analysis_progress(task_id: str):
    """Get the progress of a video analysis task"""
    if task_id in analysis_progress:
        return {"progress": analysis_progress[task_id]}
    return {"error": "Task not found"}

@app.post("/api/analyze")
async def analyze_videos(video_ids: List[str], background_tasks: BackgroundTasks):
    """Analyze selected videos using instructor and litellm with parallel processing"""
    task_id = f"task_{len(video_ids)}_{int(time.time())}"
    analysis_progress[task_id] = 0
    results = []

    try:
        # Check cache first
        videos_to_process = []
        for video_id in video_ids:
            if video_id in analysis_cache:
                results.append(analysis_cache[video_id])
            else:
                videos_to_process.append(video_id)

        if not videos_to_process:
            return {"task_id": task_id, "results": results, "status": "completed"}

        # Initialize Kaltura session
        if not init_kaltura_session():
            analysis_progress[task_id] = -1  # Indicate error
            return {
                "error": "Failed to initialize Kaltura session. Please check your credentials.",
                "task_id": task_id,
                "status": "failed"
            }

        async def process_chunk(chunk: dict, chunk_index: int, total_chunks: int, video_duration: float) -> VideoAnalysis:
            """Process a single chunk of transcript"""
            try:
                # Format transcript with timestamps
                timestamped_lines = []
                chunk_text = []
                for entry in chunk['entries']:
                    start_time = entry['startTime'] / 1000  # Convert to seconds
                    end_time = entry['endTime'] / 1000
                    timestamped_lines.append(f"[{start_time:.1f}s - {end_time:.1f}s] {entry['text']}")
                    chunk_text.append(entry['text'])

                chunk_start = chunk['start']
                chunk_end = chunk['end']
                
                print(f"Processing chunk {chunk_index + 1}/{total_chunks} ({len(''.join(chunk_text))} characters)")
                
                prompt = f"""Analyze this video transcript section ({chunk_index + 1} of {total_chunks}), covering time range {chunk_start}-{chunk_end} seconds.
                
                Transcript:
                Each line below includes its exact start and end timestamps:
                {chr(10).join(timestamped_lines)}
                
                Instructions for Analysis:
                1. Provide a well-structured summary using markdown formatting:
                   - Use # for main headings
                   - Use ## for subheadings
                   - Use bullet points (•) for lists
                   - Use proper paragraphs with line breaks
                   - Highlight key terms in **bold**
                2. Focus on the core content and main ideas
                3. Identify overarching themes and concepts
                
                Format your response as a VideoAnalysis object with:
                - summary: A well-formatted summary using markdown syntax for structure
                - insights: List of key takeaways (each prefixed with •)
                - topics: List of main topics with importance scores (1-5)
                - timestamps: [] (leave empty as timestamps will be generated separately)

Example structure:
{{
  "summary": "Clear summary of main points...",
  "insights": ["Key insight 1", "Key insight 2"],
  "topics": [
    {{"name": "Topic 1", "importance": 4}},
    {{"name": "Topic 2", "importance": 3}}
  ],
  "timestamps": []  # Timestamps will be generated separately
}}"""

                print(f"Sending chunk {chunk_index + 1} to LLM for analysis...")
                analysis = await asyncio.to_thread(
                    llm_client.chat.completions.create,
                    model=os.getenv('MODEL_ID', 'anthropic.claude-3-sonnet-20240229-v1:0'),
                    response_model=VideoAnalysis,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=MODEL_TIMEOUT,
                    max_tokens=MODEL_MAX_TOKENS,
                    temperature=MODEL_TEMPERATURE
                )
                return analysis
            except Exception as e:
                error_msg = f"Error processing chunk {chunk_index}: {str(e)}"
                print(f"ERROR: {error_msg}")
                print(f"Stack trace: {e.__traceback__}")
                error_analysis = VideoAnalysis(
                    summary=f"Error processing chunk {chunk_index}",
                    insights=[f"Error: {str(e)}"],
                    topics=[{"name": "<ERROR>", "importance": 1}],
                    timestamps=[TimestampEntry(
                        start_timestamp=0,
                        end_timestamp=0,
                        description="Processing error",
                        topic="Error",
                        importance=1
                    )]
                )
                return error_analysis

        async def process_video(video_id: str, task_id: str, total_videos: int) -> dict:
            """Process a single video with parallel chunk processing"""
            try:
                # Get English captions
                captions = kaltura.get_english_captions(video_id)
                if not captions:
                    return {
                        "video_id": video_id,
                        "analysis": {
                            "summary": "No English captions found for this video.",
                            "insights": ["Video has no English captions available"],
                            "topics": [{"name": "<NO_CAPTIONS>", "importance": 1}],
                            "timestamps": [{"start_timestamp": 0, "end_timestamp": 0, "description": "No captions available", "topic": "Error", "importance": 1}]
                        }
                    }

                # Get JSON transcript and chunk it
                print(f"Getting JSON transcript for video {video_id}...")
                transcript_chunks = kaltura.get_json_transcript(captions[0]['id'])
                if not transcript_chunks:
                    return {
                        "video_id": video_id,
                        "analysis": {
                            "summary": "No readable transcript content found.",
                            "insights": ["Video transcript is empty or unreadable"],
                            "topics": [{"name": "<NO_TRANSCRIPT>", "importance": 1}],
                            "timestamps": [{"start_timestamp": 0, "end_timestamp": 0, "description": "No transcript available", "topic": "Error", "importance": 1}]
                        }
                    }

                print(f"Got {len(transcript_chunks)} transcript chunks")
                
                # Get video duration from Kaltura
                video_info = kaltura.client.media.get(video_id)
                video_duration = float(video_info.duration)
                
                # Process chunks in parallel
                chunk_tasks = []
                for i, chunk in enumerate(transcript_chunks):
                    # Preserve individual entries with their timestamps
                    chunk_data = {
                        "entries": chunk,  # Original entries with timestamps
                        "start": chunk[0]['startTime'] / 1000,  # Convert ms to seconds
                        "end": chunk[-1]['endTime'] / 1000  # Convert ms to seconds
                    }
                    chunk_tasks.append(process_chunk(chunk_data, i, len(transcript_chunks), video_duration))

                print(f"Starting parallel analysis of {len(chunk_tasks)} chunks...")
                chunk_analyses = await asyncio.gather(*chunk_tasks)

                # Generate timestamps from full transcript
                print("Generating timestamps from full transcript...")
                timestamps = await generate_timestamps(transcript_chunks, video_duration)
                print(f"Generated {len(timestamps)} timestamps")

                # Combine analyses
                print(f"Combining analyses from {len(chunk_analyses)} chunks...")
                final_analysis = finalize_segments(chunk_analyses, video_duration)
                
                # Add timestamps to final analysis
                final_analysis["timestamps"] = timestamps
                
                result = {
                    "video_id": video_id,
                    "analysis": final_analysis
                }

                # Update progress
                analysis_progress[task_id] = min(
                    ((len(results) + 1) / total_videos) * 100,
                    99  # Keep at 99% until fully complete
                )

                # Cache the result
                analysis_cache[video_id] = result
                return result

            except Exception as e:
                print(f"Error processing video {video_id}: {str(e)}")
                return {
                    "video_id": video_id,
                    "analysis": {
                        "summary": "Error processing video.",
                        "insights": [f"Processing error: {str(e)}"],
                        "topics": [{"name": "<ERROR>", "importance": 1}],
                        "timestamps": [{"start_timestamp": 0, "end_timestamp": 0, "description": "Processing error", "topic": "Error", "importance": 1}]
                    }
                }

        # Process videos in parallel
        total_videos = len(videos_to_process)
        print(f"\nStarting analysis of {total_videos} videos...")
        video_tasks = [
            process_video(
                video_id=video_id,
                task_id=task_id,
                total_videos=total_videos
            ) for video_id in videos_to_process
        ]
        
        # Process all videos and collect results
        video_results = await asyncio.gather(*video_tasks)
        results.extend(video_results)

        # Update final progress and return results
        analysis_progress[task_id] = 100
        return {
            "task_id": task_id,
            "results": results,
            "status": "completed"
        }

    except Exception as e:
        print(f"Fatal error in analyze_videos: {str(e)}")
        analysis_progress[task_id] = -1  # Indicate error
        return {
            "error": str(e),
            "task_id": task_id,
            "status": "failed"
        }

class ChatRequest(BaseModel):
    question: str
    context: List[dict]

@app.post("/api/chat")
async def chat_with_videos(request: ChatRequest):
    """Chat with the analyzed videos using instructor and litellm"""
    try:
        print(f"Received chat request with context from {len(request.context)} videos")
        
        prompt = f"""Question: {request.question}

Video Context:
{json.dumps(request.context, indent=2)}

Instructions:
Provide a clear, direct answer based on the video context above. Focus on accuracy and relevance."""
        
        print("Sending chat request to LLM...")
        response = await asyncio.to_thread(
            llm_client.chat.completions.create,
            model=os.getenv('MODEL_ID', 'anthropic.claude-3-sonnet-20240229-v1:0'),
            response_model=ChatResponse,
            messages=[{"role": "user", "content": prompt}],
            temperature=MODEL_TEMPERATURE
        )
        
        return {"answer": response.answer}
    except asyncio.TimeoutError as e:
        print(f"Timeout error in chat: {e}")
        return {"error": "Request timed out. Please try again."}
    except Exception as e:
        print(f"Unexpected error in chat: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv('HOST', '0.0.0.0'),
        port=int(os.getenv('PORT', 8000)),
        reload=True,
        server_header=False,
        proxy_headers=True
    )