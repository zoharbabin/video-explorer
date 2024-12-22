# pylint: disable=no-member
import os
import json
import time
import math
import asyncio
import logging
from logger_config import setup_logging

# Set up logging configuration
setup_logging(os.getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)
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
        logger.error("Failed to initialize Kaltura session")
        return False
    logger.info(f"Successfully initialized Kaltura session for partner ID: {pid}")
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
        logger.info("Initializing Kaltura session...")
        if not init_kaltura_session():
            logger.error("Failed to initialize Kaltura session")
            return {"error": "Failed to initialize Kaltura session. Please check your credentials."}
        
        logger.info("Session initialized successfully")
        
        # Use the new fetch_videos method that ensures videos have captions
        videos = kaltura.fetch_videos(
            category_ids=category_id,
            free_text=query,
            number_of_videos=PAGE_SIZE
        )
        
        logger.info(f"Found {len(videos)} videos with captions")
        return {"videos": videos}
        
    except Exception as e:
        logger.error(f"Error in search_videos: {str(e)}", exc_info=True)
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
        logger.debug(f"Processing segment {segment_index + 1}/{total_segments}")
        logger.debug(f"Input chunks: {len(segment_chunks)} chunks with {sum(len(chunk) for chunk in segment_chunks)} total entries")
        
        # --------------------------------------------------------------------
        # 1. Determine start/end times of this segment
        # --------------------------------------------------------------------
        segment_start = round(segment_chunks[0][0]['startTime'] / 1000, 2)
        segment_end   = round(segment_chunks[-1][-1]['endTime'] / 1000, 2)

        # If the segment is extremely short, we can process anyway,
        # but optionally warn (instead of skipping).
        segment_length = segment_end - segment_start
        logger.debug(f"Segment timing: {segment_start:.1f}s -> {segment_end:.1f}s (duration: {segment_length:.1f}s)")
        if segment_length < 15:
            logger.warning(f"Very short segment (only {segment_length:.2f} seconds).")

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
8. Each topic should be at least 30 seconds long.

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
        logger.debug("Calling LLM for segment analysis...")
        segment_timestamps = await asyncio.to_thread(
            llm_client.chat.completions.create,
            model=os.getenv('MODEL_ID', 'anthropic.claude-3-sonnet-20240229-v1:0'),
            response_model=List[TimestampEntry],
            messages=[{"role": "user", "content": prompt}],
            timeout=MODEL_TIMEOUT,
            max_tokens=MODEL_MAX_TOKENS,
            temperature=MODEL_TEMPERATURE
        )
        logger.debug(f"LLM returned {len(segment_timestamps)} timestamps")

        # --------------------------------------------------------------------
        # 5. Validate the LLM's Results
        # --------------------------------------------------------------------
        logger.debug(f"Starting validation of {len(segment_timestamps)} timestamps")
        valid_results = []
        used_timestamps = set()

        for ts in segment_timestamps:
            logger.debug(f"Validating timestamp: {ts.start_timestamp:.1f}s - {ts.end_timestamp:.1f}s")
            logger.debug(f"Description: {ts.description}")
            logger.debug(f"Topic: {ts.topic}")
            logger.debug(f"Importance: {ts.importance}")
            
            # Ensure timestamps are within segment range AND video duration
            if ts.start_timestamp < (segment_start - 0.1):
                logger.debug(f"❌ Start timestamp {ts.start_timestamp:.1f}s before segment start [{segment_start:.1f}s]")
                continue
            if ts.end_timestamp > (segment_end + 0.1):
                logger.debug(f"❌ End timestamp {ts.end_timestamp:.1f}s after segment end [{segment_end:.1f}s]")
                continue
            if ts.end_timestamp > (video_duration + 0.1):
                logger.debug(f"❌ End timestamp {ts.end_timestamp:.1f}s exceeds video duration {video_duration:.1f}s")
                continue
            if ts.start_timestamp >= ts.end_timestamp:
                logger.debug(f"❌ Start timestamp {ts.start_timestamp:.1f}s not before end timestamp {ts.end_timestamp:.1f}s")
                continue

            # If we've already used this exact timestamp pair, skip
            timestamp_pair = (ts.start_timestamp, ts.end_timestamp)
            if timestamp_pair in used_timestamps:
                logger.debug(f"❌ Duplicate timestamp pair {ts.start_timestamp:.1f}s - {ts.end_timestamp:.1f}s")
                continue

            # Good enough to accept
            logger.debug(f"✓ Accepted timestamp pair {ts.start_timestamp:.1f}s - {ts.end_timestamp:.1f}s")
            used_timestamps.add(timestamp_pair)
            valid_results.append(ts)

        logger.debug("Final validation results:")
        logger.debug(f"- Input timestamps: {len(segment_timestamps)}")
        logger.debug(f"- Valid timestamps: {len(valid_results)}")
        logger.debug(f"- Rejected timestamps: {len(segment_timestamps) - len(valid_results)}")
        
        # Return the final, validated timestamps
        return valid_results

    except Exception as e:
        logger.error(f"❌ ERROR processing segment {segment_index + 1}:")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error("Segment details:")
        logger.error(f"- Start time: {segment_start:.1f}s")
        logger.error(f"- End time: {segment_end:.1f}s")
        logger.error(f"- Duration: {segment_length:.1f}s")
        logger.error(f"- Number of chunks: {len(segment_chunks)}")
        logger.error(f"- Total entries: {sum(len(chunk) for chunk in segment_chunks)}")
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
        logger.info(f"Processing {len(segment_tasks)} transcript segments for timestamps...")
        segment_results = await asyncio.gather(*segment_tasks)
        
        # Combine all timestamps
        all_timestamps = [ts for result in segment_results for ts in result]
        logger.debug(f"Combined {len(all_timestamps)} timestamps from all segments")
        
        # Sort and filter timestamps with minimum spacing
        filtered_timestamps = []
        min_gap = 30  # 30 seconds minimum gap
        
        for ts in sorted(all_timestamps, key=lambda x: x.start_timestamp):
            # Validate timestamps are within video duration
            if ts.end_timestamp > video_duration:
                logger.debug(f"Skipping timestamp {ts.end_timestamp:.1f}s that exceeds video duration {video_duration:.1f}s")
                continue
                
            # Check spacing
            if not filtered_timestamps:
                filtered_timestamps.append(ts)
                logger.debug(f"Added first timestamp: {ts.start_timestamp:.1f}s - {ts.end_timestamp:.1f}s")
            elif (ts.start_timestamp - filtered_timestamps[-1].end_timestamp) >= min_gap:
                filtered_timestamps.append(ts)
                logger.debug(f"Added timestamp with sufficient gap: {ts.start_timestamp:.1f}s - {ts.end_timestamp:.1f}s")
        
        # Ensure reasonable coverage
        target_count = int(video_duration / 180)  # Aim for one timestamp every ~3 minutes
        if len(filtered_timestamps) < target_count * 0.7:  # If we have less than 70% of target
            logger.warning(f"Generated only {len(filtered_timestamps)} timestamps for {video_duration/60:.1f} minute video")
        
        logger.info(f"Final timestamp count: {len(filtered_timestamps)}")
        return filtered_timestamps

    except Exception as e:
        logger.error(f"Error generating timestamps: {str(e)}", exc_info=True)
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
                logger.info(f"Fetching English captions for video {video_id}...")
                captions = kaltura.get_english_captions(video_id)
                if not captions:
                    logger.warning(f"No English captions found for video {video_id}")
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
                logger.info(f"Getting JSON transcript for video {video_id}...")
                transcript_chunks = kaltura.get_json_transcript(captions[0]['id'])
                if not transcript_chunks:
                    logger.warning(f"No readable transcript content found for video {video_id}")
                    return {
                        "video_id": video_id,
                        "analysis": {
                            "summary": "No readable transcript content found.",
                            "insights": ["Video transcript is empty or unreadable"],
                            "topics": [{"name": "<NO_TRANSCRIPT>", "importance": 1}],
                            "timestamps": [{"start_timestamp": 0, "end_timestamp": 0, "description": "No transcript available", "topic": "Error", "importance": 1}]
                        }
                    }

                logger.info(f"Got {len(transcript_chunks)} transcript chunks for video {video_id}")
                
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

                logger.info(f"Starting parallel analysis of {len(chunk_tasks)} chunks...")
                chunk_analyses = await asyncio.gather(*chunk_tasks)
                # Combine analyses first
                logger.info(f"Combining analyses from {len(chunk_analyses)} chunks...")
                final_analysis = finalize_segments(chunk_analyses, video_duration)

                # Generate timestamps from full transcript
                logger.info("Generating timestamps from full transcript...")
                timestamps = await generate_timestamps(transcript_chunks, video_duration)
                logger.info(f"Generated {len(timestamps)} timestamps")

                # Add timestamps to final analysis
                final_analysis["timestamps"] = timestamps

                # Now generate social clips with timestamps available
                logger.info("Generating social clips suggestions...")
                social_clips = await generate_social_clips(final_analysis)
                final_analysis["social_clips"] = social_clips
                logger.info(f"Generated {len(social_clips)} social clip suggestions")
                
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
                logger.error(f"Error processing video {video_id}: {str(e)}", exc_info=True)
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
        logger.info(f"Starting analysis of {total_videos} videos...")
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
        logger.error(f"Fatal error in analyze_videos: {str(e)}", exc_info=True)
        analysis_progress[task_id] = -1  # Indicate error
        return {
            "error": str(e),
            "task_id": task_id,
            "status": "failed"
        }

class SocialClipSuggestion(BaseModel):
    start_time: float = Field(description="Start time of the clip in seconds")
    end_time: float = Field(description="End time of the clip in seconds")
    description: str = Field(description="Description for the social media post")
    hashtags: str = Field(description="Suggested hashtags for the post")
    platform: str = Field(description="Target platform (LinkedIn or YouTube)")

class SocialClipsResponse(BaseModel):
    suggestions: List[SocialClipSuggestion]

class ChatRequest(BaseModel):
    question: str
    context: List[dict]

async def generate_social_clips(context: dict) -> List[SocialClipSuggestion]:
    """Generate suggestions for social media clips"""
    try:
        logger.info("Generating social clip suggestions...")
        
        logger.debug("Analyzing context for social clips...")
        logger.debug(f"Context type: {type(context)}")
        logger.debug(f"Context keys: {list(context.keys())}")
        
        # Extract and validate key moments and topics
        key_moments = context.get('timestamps', [])
        logger.debug(f"Raw key_moments: {key_moments}")
        logger.debug(f"Type of key_moments: {type(key_moments)}")
        
        if not key_moments:
            logger.warning("❌ No key moments found in context")
            return []
            
        # Ensure timestamps are in the correct format
        formatted_moments = []
        logger.debug("Processing timestamps:")
        for moment in key_moments:
            try:
                # If it's a dict, ensure it has the required fields
                if isinstance(moment, dict):
                    if all(k in moment for k in ['start_timestamp', 'end_timestamp', 'description', 'topic', 'importance']):
                        formatted_moments.append(moment)
                        logger.debug(f"✓ Valid timestamp: {moment['start_timestamp']}s - {moment['end_timestamp']}s ({moment['topic']})")
                    else:
                        logger.warning(f"❌ Dict missing required fields: {moment}")
                        continue
                
                # If it's a TimestampEntry, convert to dict
                elif isinstance(moment, TimestampEntry):
                    moment_dict = {
                        'start_timestamp': moment.start_timestamp,
                        'end_timestamp': moment.end_timestamp,
                        'description': moment.description,
                        'topic': moment.topic,
                        'importance': moment.importance
                    }
                    formatted_moments.append(moment_dict)
                    logger.debug(f"✓ Converted TimestampEntry: {moment.start_timestamp}s - {moment.end_timestamp}s ({moment.topic})")
                
                # If it has the required attributes, create a dict
                elif all(hasattr(moment, attr) for attr in ['start_timestamp', 'end_timestamp', 'description', 'topic', 'importance']):
                    moment_dict = {
                        'start_timestamp': getattr(moment, 'start_timestamp'),
                        'end_timestamp': getattr(moment, 'end_timestamp'),
                        'description': getattr(moment, 'description'),
                        'topic': getattr(moment, 'topic'),
                        'importance': getattr(moment, 'importance')
                    }
                    formatted_moments.append(moment_dict)
                    logger.debug(f"✓ Converted object: {moment_dict['start_timestamp']}s - {moment_dict['end_timestamp']}s ({moment_dict['topic']})")
                
                else:
                    logger.warning(f"❌ Invalid timestamp format: {type(moment)}")
                    logger.debug(f"Available attributes: {dir(moment) if hasattr(moment, '__dir__') else 'No attributes'}")
                    continue
                    
            except Exception as e:
                logger.error(f"❌ Error processing timestamp: {str(e)}", exc_info=True)
                continue
        
        if not formatted_moments:
            logger.warning("❌ No valid formatted moments")
            return []
            
        # Replace the key_moments with properly formatted ones
        key_moments = formatted_moments
        logger.info(f"✓ Successfully formatted {len(key_moments)} timestamps")
            
        # Ensure topics are JSON serializable
        topics = []
        raw_topics = context.get('topics', [])
        for topic in raw_topics:
            if isinstance(topic, dict):
                topics.append(topic)
            else:
                topics.append({
                    'name': topic.name if hasattr(topic, 'name') else str(topic),
                    'importance': topic.importance if hasattr(topic, 'importance') else 3
                })
        
        if not topics:
            print("⚠️ No topics found in context, will generate generic hashtags")
            
        print(f"Found {len(key_moments)} key moments and {len(topics)} topics")
        
        # Format timestamps for the prompt, handling both dict and TimestampEntry objects
        formatted_timestamps = []
        for moment in key_moments:
            if isinstance(moment, dict):
                formatted_timestamps.append({
                    'start_time': moment['start_timestamp'],
                    'end_time': moment['end_timestamp'],
                    'description': moment['description'],
                    'topic': moment['topic'],
                    'importance': moment['importance']
                })
            else:  # TimestampEntry object
                formatted_timestamps.append({
                    'start_time': moment.start_timestamp,
                    'end_time': moment.end_timestamp,
                    'description': moment.description,
                    'topic': moment.topic,
                    'importance': moment.importance
                })

        # Prepare example without f-string interpolation
        example = {
            "suggestions": [
                {
                    "start_time": 120.5,
                    "end_time": 180.0,
                    "description": "🔥 Controversial take: Traditional machine learning is dead. Watch how our new approach achieves 95% accuracy while completely bypassing conventional neural networks. Do you agree this is the future?",
                    "hashtags": "#AIDebate #FutureOfML #TechDisruption #Innovation",
                    "platform": "LinkedIn"
                }
            ]
        }

        prompt = f"""You are a social media expert tasked with identifying viral-worthy clips from video content. Your goal is to find moments that will spark intense discussions and debates on LinkedIn and YouTube.

AVAILABLE TIMESTAMPS AND TOPICS:
IMPORTANT: Use timestamps from the AVAILABLE TIMESTAMPS FOR CLIPS section below. Each clip MUST use exact start_time and end_time values from this list:
{json.dumps(formatted_timestamps, indent=2)}

Main Topics and Their Importance:
{json.dumps(topics, indent=2)}

TASK:
Identify 3-4 highly engaging clips that will generate maximum engagement on social media. Each clip must:
1. Use EXACT start_time and end_time from the Key Moments list above
2. Be between 30-300 seconds in duration (0.5 to 5 minutes)
3. Focus on controversial or debate-worthy content
4. Include a provocative description that encourages responses
5. Use trending hashtags relevant to the topic

CLIP SELECTION CRITERIA:
1. Controversial Content:
   - Challenges established industry practices
   - Presents unexpected or surprising results
   - Offers contrarian viewpoints
   - Questions common assumptions

2. Platform-Specific Focus:
   - LinkedIn: Industry disruption, professional debates, future trends
   - YouTube: Technical deep-dives, visual demonstrations, broader discussions

3. Viral Potential:
   - Emotionally engaging content
   - "Hot take" moments
   - Surprising revelations
   - Counterintuitive findings

YOUR RESPONSE MUST BE A VALID JSON OBJECT WITH THIS EXACT STRUCTURE:
{json.dumps({"suggestions": [
    {
        "start_time": 123.4,  # MUST match a timestamp from Key Moments
        "end_time": 189.7,    # MUST match a timestamp from Key Moments
        "description": "🔥 Hot Take: Why traditional approaches are failing...",
        "hashtags": "#TrendingHashtag #Industry #Topic",
        "platform": "LinkedIn"  # or "YouTube"
    }
]}, indent=2)}

IMPORTANT:
- Only use timestamps that exist in the Key Moments list
- Ensure descriptions are provocative and encourage debate
- Include trending hashtags
- Format must match the example exactly"""
        
        # Create a JSON-serializable version of the context
        serializable_context = {
            'summary': context.get('summary', ''),
            'insights': context.get('insights', []),
            'topics': topics,  # We already made this serializable
            'timestamps': formatted_timestamps  # We already made this serializable
        }
        
        # Add serializable context to prompt
        prompt += f"\n\nVideo Context:\n{json.dumps(serializable_context, indent=2)}"
        
        logger.info("Sending social clips request to LLM...")
        try:
            # Get response with proper model
            response = await asyncio.to_thread(
                llm_client.chat.completions.create,
                model=os.getenv('MODEL_ID', 'anthropic.claude-3-sonnet-20240229-v1:0'),
                response_model=SocialClipsResponse,
                messages=[{"role": "user", "content": prompt}],
                temperature=MODEL_TEMPERATURE
            )
            
            if not response or not response.suggestions:
                logger.warning("❌ No valid suggestions in LLM response")
                return []
                
            logger.info(f"✓ Successfully parsed {len(response.suggestions)} suggestions from LLM")
            
        except Exception as e:
            logger.error(f"❌ Error in LLM request: {str(e)}", exc_info=True)
            return []
        
        # Validate and filter suggestions
        valid_suggestions = []
        used_timestamps = set()
        
        logger.info(f"Validating {len(response.suggestions)} social clip suggestions...")
        for suggestion in response.suggestions:
            logger.debug("Validating clip suggestion:")
            logger.debug(f"Start time: {suggestion.start_time:.1f}s")
            logger.debug(f"End time: {suggestion.end_time:.1f}s")
            logger.debug(f"Platform: {suggestion.platform}")
            logger.debug(f"Description: {suggestion.description}")
            
            # Find best matching key moment
            best_match = None
            min_diff = float('inf')
            
            for moment in context.get('timestamps', []):
                # Get start and end timestamps regardless of format
                moment_start = moment.start_timestamp if hasattr(moment, 'start_timestamp') else moment.get('start_timestamp', 0)
                moment_end = moment.end_timestamp if hasattr(moment, 'end_timestamp') else moment.get('end_timestamp', 0)
                
                start_diff = abs(suggestion.start_time - moment_start)
                end_diff = abs(suggestion.end_time - moment_end)
                total_diff = start_diff + end_diff
                
                if total_diff < min_diff:
                    min_diff = total_diff
                    best_match = moment
            
            # Accept if within 60 seconds total tolerance for longer segments
            if best_match and min_diff <= 60:  # Allow up to 30 seconds difference per timestamp
                logger.debug(f"✓ Found matching key moment: {best_match.start_timestamp:.1f}s - {best_match.end_timestamp:.1f}s")
                # Use exact timestamps from the key moment, handling both object and dict formats
                suggestion.start_time = best_match.start_timestamp if hasattr(best_match, 'start_timestamp') else best_match.get('start_timestamp')
                suggestion.end_time = best_match.end_timestamp if hasattr(best_match, 'end_timestamp') else best_match.get('end_timestamp')
            else:
                logger.debug(f"❌ No close matching key moment found (min diff: {min_diff:.1f}s)")
                continue
                
            # Validate clip duration with high flexibility for longer segments
            duration = suggestion.end_time - suggestion.start_time
            min_duration = 20  # Minimum 20 seconds
            max_duration = 300  # Allow up to 5 minutes for important segments
            
            logger.debug(f"Checking duration: {duration:.1f}s")
            if duration < min_duration:
                logger.debug(f"❌ Clip too short: {duration:.1f}s < {min_duration}s minimum")
                continue
            if duration > max_duration:
                logger.debug(f"❌ Clip too long: {duration:.1f}s > {max_duration}s maximum")
                continue
                
            logger.debug(f"✓ Duration acceptable: {duration:.1f}s")
                
            # Check for overlapping clips
            timestamp_pair = (suggestion.start_time, suggestion.end_time)
            if timestamp_pair in used_timestamps:
                logger.debug(f"❌ Skipping clip: Duplicate timestamp pair {suggestion.start_time:.1f}s - {suggestion.end_time:.1f}s")
                continue
                
            # Enhance description for better engagement
            logger.debug("Checking description engagement...")
            engagement_markers = ['?', 'agree', 'think', 'debate', 'discuss', 'share', 'opinion', 'controversial', 'surprising']
            provocative_markers = ['🔥', '💡', '🤔', '👀', '💪', '🚀']
            
            has_engagement = any(marker in suggestion.description.lower() for marker in engagement_markers)
            has_emoji = any(marker in suggestion.description for marker in provocative_markers)
            
            if not has_engagement or not has_emoji:
                logger.debug("Enhancing description engagement...")
                if not has_emoji:
                    suggestion.description = f"🔥 {suggestion.description}"
                if not has_engagement:
                    suggestion.description += " What's your take on this? Share your thoughts! 💭"
                logger.debug(f"✓ Enhanced description: {suggestion.description}")
            else:
                logger.debug("✓ Description already engaging")
                
            # Enhance hashtags for better reach
            logger.debug("Enhancing hashtags...")
            current_hashtags = suggestion.hashtags.split()
            
            # Get top topics by importance
            top_topics = sorted(context.get('topics', []), key=lambda x: x.get('importance', 0), reverse=True)[:3]
            topic_hashtags = [f"#{topic['name'].replace(' ', '')}" for topic in top_topics]
            
            # Add trending and topic-specific hashtags
            trending_hashtags = ['#TrendingNow', '#MustWatch', '#Innovation']
            platform_hashtags = {
                'LinkedIn': ['#Leadership', '#Innovation', '#FutureOfWork', '#ProfessionalDevelopment'],
                'YouTube': ['#Tutorial', '#HowTo', '#LearnOnYouTube', '#Education']
            }
            
            # Combine all hashtags and ensure uniqueness
            all_hashtags = set(current_hashtags + topic_hashtags + trending_hashtags + platform_hashtags.get(suggestion.platform, []))
            
            # Select top 8 hashtags, prioritizing topic and platform-specific ones
            final_hashtags = (topic_hashtags +
                            platform_hashtags.get(suggestion.platform, [])[:2] +
                            list(all_hashtags))[:8]
            
            suggestion.hashtags = ' '.join(sorted(set(final_hashtags)))  # Remove any duplicates
            logger.debug(f"✓ Enhanced hashtags: {suggestion.hashtags}")
                
            logger.debug(f"✓ Valid clip: {suggestion.start_time:.1f}s - {suggestion.end_time:.1f}s")
            used_timestamps.add(timestamp_pair)
            valid_suggestions.append(suggestion)
            
        logger.info(f"Found {len(valid_suggestions)} valid social clip suggestions")
        return valid_suggestions
    except Exception as e:
        logger.error(f"Error generating social clips: {e}", exc_info=True)
        return []

@app.post("/api/chat")
async def chat_with_videos(request: ChatRequest):
    """Chat with the analyzed videos using instructor and litellm"""
    try:
        logger.info(f"Received chat request with context from {len(request.context)} videos")
        
        prompt = f"""Question: {request.question}

Video Context:
{json.dumps(request.context, indent=2)}

Instructions:
Provide a clear, direct answer based on the video context above. Focus on accuracy and relevance."""
        
        logger.info("Sending chat request to LLM...")
        response = await asyncio.to_thread(
            llm_client.chat.completions.create,
            model=os.getenv('MODEL_ID', 'anthropic.claude-3-sonnet-20240229-v1:0'),
            response_model=ChatResponse,
            messages=[{"role": "user", "content": prompt}],
            temperature=MODEL_TEMPERATURE
        )
        
        logger.info("Successfully received LLM response")
        return {"answer": response.answer}
    except asyncio.TimeoutError as e:
        logger.error(f"Timeout error in chat: {e}")
        return {"error": "Request timed out. Please try again."}
    except Exception as e:
        logger.error(f"Unexpected error in chat: {e}", exc_info=True)
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