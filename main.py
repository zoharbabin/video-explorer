# pylint: disable=no-member
import os
import json
import time
import asyncio
import aiohttp
import re
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
from KalturaClient import *
from KalturaClient.Plugins.Core import (
    KalturaSessionType,
    KalturaMediaEntryFilter,
    KalturaMediaType,
    KalturaFilterPager,
    KalturaMediaEntryOrderBy
)
from KalturaClient.Plugins.Caption import KalturaCaptionAssetFilter

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

# Initialize Kaltura configuration
config = KalturaConfiguration()
config.serviceUrl = os.getenv('KALTURA_SERVICE_URL', 'https://cdnapisec.kaltura.com')  # Use secure API endpoint
print(f"Initializing Kaltura client with service URL: {config.serviceUrl}")  # Debug log
client = KalturaClient(config)

# Initialize instructor client with litellm
llm_client = instructor.from_litellm(completion)

# Configure litellm to use Bedrock
os.environ["AWS_REGION"] = os.getenv('AWS_REGION', 'us-east-1')

# Load configuration values
KALTURA_SESSION_DURATION = int(os.getenv('KALTURA_SESSION_DURATION', 86400))  # 24 hours
PAGE_SIZE = int(os.getenv('PAGE_SIZE', 10))
MODEL_TIMEOUT = int(os.getenv('MODEL_TIMEOUT', 60))
MODEL_MAX_TOKENS = int(os.getenv('MODEL_MAX_TOKENS', 4000))
MODEL_CHUNK_SIZE = int(os.getenv('MODEL_CHUNK_SIZE', 24000))  # Doubled chunk size for fewer API calls
MODEL_TEMPERATURE = float(os.getenv('MODEL_TEMPERATURE', 0))

# Define Pydantic models for structured outputs
class TimestampEntry(BaseModel):
    timestamp: float = Field(description="Timestamp in seconds")
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
    try:
        partner_id = int(os.getenv('KALTURA_PARTNER_ID'))
        secret = os.getenv('KALTURA_SECRET')
        
        print(f"Attempting to initialize Kaltura session with partner ID: {partner_id}")  # Debug log
        
        if not partner_id or not secret:
            print("Missing Kaltura credentials in environment variables")
            return False
            
        try:
            session = client.session.start(
                secret,
                None,
                KalturaSessionType.ADMIN,
                partner_id,
                KALTURA_SESSION_DURATION,
                "appid:video-explorer"
            )
            print(f"Session created successfully: {session[:30]}...")  # Debug log (show first 30 chars of session)
            client.setKs(session)
            
            # Verify session by making a test API call
            try:
                test_filter = KalturaMediaEntryFilter()
                test_filter.mediaTypeEqual = KalturaMediaType.VIDEO
                test_pager = KalturaFilterPager()
                test_pager.pageSize = 1
                test_result = client.media.list(test_filter, test_pager)
                print(f"Test API call successful. Total count: {test_result.totalCount}")  # Debug log
                return True
            except Exception as api_error:
                print(f"Session created but test API call failed: {api_error}")
                return False
                
        except Exception as session_error:
            print(f"Failed to create Kaltura session: {session_error}")
            return False
            
    except Exception as e:
        print(f"Failed to initialize Kaltura session: {e}")
        return False

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
        print("Initializing Kaltura session...")  # Debug log
        if not init_kaltura_session():
            return {"error": "Failed to initialize Kaltura session. Please check your credentials."}
        
        print("Session initialized successfully")  # Debug log
            
        # Create filter for video search
        filter = KalturaMediaEntryFilter()
        filter.mediaTypeEqual = KalturaMediaType.VIDEO
        
        if category_id:
            filter.categoryAncestorIdIn = category_id
        if query:
            filter.freeText = query
        else:
            # When no search parameters are provided, sort by creation date (newest first)
            filter.orderBy = KalturaMediaEntryOrderBy.CREATED_AT_DESC
            
        # Get videos with captions
        pager = KalturaFilterPager()
        pager.pageSize = PAGE_SIZE
        videos = []
        
        try:
            print("Executing Kaltura media.list API call...")  # Debug log
            result = client.media.list(filter, pager)
            print(f"Found {result.totalCount} total videos")  # Debug log
            
            if not result.objects:
                print("No videos returned from the API")  # Debug log
                return {"videos": []}
        except Exception as api_error:
            print(f"Error executing media.list API call: {api_error}")  # Debug log
            return {"error": f"Failed to fetch videos: {str(api_error)}"}
            
        for entry in result.objects:
            # Include all videos for now, without caption check
            videos.append({
                "id": entry.id,
                "name": entry.name,
                "description": entry.description,
                "duration": entry.duration,
                "thumbnail_url": entry.thumbnailUrl
            })
            print(f"Added video: {entry.name}")  # Debug log
                
        return {"videos": videos}
    except Exception as e:
        return {"error": str(e)}

def parse_srt_timestamp(timestamp_str: str) -> float:
    """Convert SRT timestamp to seconds"""
    # Remove milliseconds if present
    if ',' in timestamp_str:
        timestamp_str = timestamp_str.split(',')[0]
    
    # Parse HH:MM:SS format
    try:
        h, m, s = map(int, timestamp_str.split(':'))
        return h * 3600 + m * 60 + s
    except:
        return 0.0

def parse_captions_with_timestamps(srt_content: str) -> List[dict]:
    """Parse SRT content into a list of caption entries with timestamps"""
    entries = []
    current_entry = {}
    
    # Split content into blocks (separated by double newline)
    blocks = srt_content.strip().split('\n\n')
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:  # Valid SRT block should have at least 3 lines
            try:
                # Parse timestamp line (second line)
                timestamps = lines[1].split(' --> ')
                if len(timestamps) == 2:
                    start_time = parse_srt_timestamp(timestamps[0].strip())
                    end_time = parse_srt_timestamp(timestamps[1].strip())
                    
                    # Get text (remaining lines)
                    text = ' '.join(lines[2:]).strip()
                    
                    entries.append({
                        "start": start_time,
                        "end": end_time,
                        "text": text
                    })
            except Exception as e:
                print(f"Error parsing caption block: {e}")
                continue
    
    return entries

def split_transcript_by_time(
    caption_entries: List[dict],
    chunk_size_seconds: int = 600  # 10 minutes
) -> List[dict]:
    """
    Splits the transcript into time-based chunks of chunk_size_seconds.
    Each chunk contains concatenated text from all captions that fall within
    that time range.
    """
    if not caption_entries:
        return []
        
    chunks = []
    current_chunk_start = 0
    current_chunk_end = chunk_size_seconds
    current_text = []
    
    # Sort entries by start time to ensure chronological processing
    sorted_entries = sorted(caption_entries, key=lambda x: x["start"])
    
    for entry in sorted_entries:
        entry_start = entry["start"]
        entry_end = entry["end"]
        
        # If the entry is entirely before current_chunk_end, add its text
        if entry_end <= current_chunk_end:
            current_text.append(entry["text"])
        else:
            # We've reached the end of the current chunk. Save it if it has content.
            if current_text:
                chunk = {
                    "start": current_chunk_start,
                    "end": current_chunk_end,
                    "text": "\n".join(current_text).strip()
                }
                chunks.append(chunk)
            
            # Move to the next chunk boundary until this entry fits
            while entry_start >= current_chunk_end:
                current_chunk_start = current_chunk_end
                current_chunk_end += chunk_size_seconds
            
            # Start the next chunk with this entry
            current_text = [entry["text"]]
    
    # Add the final chunk if there's any leftover text
    if current_text:
        chunk = {
            "start": current_chunk_start,
            "end": current_chunk_end,
            "text": "\n".join(current_text).strip()
        }
        chunks.append(chunk)
    
    return chunks

def finalize_segments(chunk_results: List[VideoAnalysis], video_duration: float) -> dict:
    """
    Combine chunk results into a final analysis with well-spaced key moments
    """
    all_key_moments = []
    combined_summaries = []
    all_insights = []
    all_topics = []
    
    for chunk in chunk_results:
        combined_summaries.append(chunk.summary)
        all_insights.extend(chunk.insights)
        all_topics.extend(chunk.topics)
        # Filter out any timestamps that exceed video duration
        valid_timestamps = [
            ts for ts in chunk.timestamps
            if ts.timestamp <= video_duration
        ]
        all_key_moments.extend(valid_timestamps)
    
    # Sort by time
    all_key_moments.sort(key=lambda m: m.timestamp)
    
    # Simple dedup/spacing filter
    filtered_moments = []
    min_gap = 60  # 1 minute minimum gap
    
    for m in all_key_moments:
        # Double check timestamp is within video duration
        if m.timestamp > video_duration:
            continue
            
        if not filtered_moments:
            filtered_moments.append(m)
        else:
            if (m.timestamp - filtered_moments[-1].timestamp) >= min_gap:
                filtered_moments.append(m)
    
    # Ensure coverage near the end
    if filtered_moments and (video_duration - filtered_moments[-1].timestamp > 600):
        # If the last key moment is more than 10 minutes from the end,
        # we might want to add a final marker or rely on the last chunk
        pass
    
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
        "timestamps": filtered_moments
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
                chunk_text = chunk["text"]
                chunk_start = chunk["start"]
                chunk_end = chunk["end"]
                
                print(f"Processing chunk {chunk_index + 1}/{total_chunks} ({len(chunk_text)} characters)")
                
                prompt = f"""Analyze this video transcript section ({chunk_index + 1} of {total_chunks}), covering time range {chunk_start}-{chunk_end} seconds. The total video duration is {video_duration} seconds.
                
                Transcript:
                {chunk_text}
                
                Instructions:
                1. Provide a well-structured summary using proper markdown formatting:
                   - Use # for main headings
                   - Use ## for subheadings
                   - Use bullet points (•) for lists
                   - Use proper paragraphs with line breaks
                   - Highlight key terms in **bold**
                2. Identify up to 2 truly significant moments with timestamps.
                3. Only include major transitions or critical insights. Do not list minor events.
                4. IMPORTANT: All timestamps must be less than or equal to {video_duration} seconds.
                
                Format your response as a VideoAnalysis object with:
                - summary: A well-formatted summary using markdown syntax for structure
                - insights: List of key takeaways (each prefixed with •)
                - topics: List of main topics with importance scores (1-5)
                - timestamps: List of TimestampEntry objects, each with:
                  - timestamp: Time in seconds (between {chunk_start} and {chunk_end})
                  - description: Short description (8-15 words)
                  - topic: Main topic being discussed
                  - importance: Score from 1-5 (5 for pivotal moments)

Example structure:
{{
  "summary": "Clear summary of main points...",
  "insights": ["Key insight 1", "Key insight 2"],
  "topics": [
    {{"name": "Topic 1", "importance": 4}},
    {{"name": "Topic 2", "importance": 3}}
  ],
  "timestamps": [
    {{
      "timestamp": {chunk_start + 120},
      "description": "Introduces key concept with clear example",
      "topic": "Main Topic",
      "importance": 4
    }}
  ]
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
                        timestamp=0,
                        description="Processing error",
                        topic="Error",
                        importance=1
                    )]
                )
                return error_analysis

        async def process_video(video_id: str, task_id: str, total_videos: int) -> dict:
            """Process a single video with parallel chunk processing"""
            try:
                caption_filter = KalturaCaptionAssetFilter()
                caption_filter.entryIdEqual = video_id
                caption_filter.languageEqual = "en"
                captions = client.caption.captionAsset.list(caption_filter, KalturaFilterPager()).objects

                if not captions:
                    return {
                        "video_id": video_id,
                        "analysis": {
                            "summary": "No captions found for this video.",
                            "insights": ["Video has no English captions available"],
                            "topics": [{"name": "<NO_CAPTIONS>", "importance": 1}],
                            "timestamps": [{"timestamp": 0, "description": "No captions available", "topic": "Error", "importance": 1}]
                        }
                    }

                caption_url = client.caption.captionAsset.getUrl(captions[0].id)
                async with aiohttp.ClientSession() as session:
                    async with session.get(caption_url) as response:
                        caption_content = await response.text()

                cleaned_content = caption_content.replace('\x00', '').strip()
                cleaned_content = ''.join(char if ord(char) < 65536 else ' ' for char in cleaned_content)

                if not cleaned_content:
                    return {
                        "video_id": video_id,
                        "analysis": {
                            "summary": "No readable transcript content found.",
                            "insights": ["Video transcript is empty or unreadable"],
                            "topics": [{"name": "<NO_TRANSCRIPT>", "importance": 1}],
                            "timestamps": [{"timestamp": 0, "description": "No transcript available", "topic": "Error", "importance": 1}]
                        }
                    }

                # Parse captions and split into time-based chunks
                print(f"Parsing captions for video {video_id}...")
                caption_entries = parse_captions_with_timestamps(cleaned_content)
                print(f"Found {len(caption_entries)} caption entries")
                
                # Get video duration from Kaltura
                video_info = client.media.get(video_id)
                video_duration = float(video_info.duration)
                
                # Calculate optimal chunk size based on video duration
                chunk_size = min(600, max(300, video_duration / 8))  # Between 5-10 minutes
                print(f"Using chunk size of {chunk_size} seconds for {video_duration} second video")
                
                time_chunks = split_transcript_by_time(caption_entries, chunk_size_seconds=int(chunk_size))
                print(f"Split into {len(time_chunks)} time-based chunks")
                
                # Process chunks in parallel
                chunk_tasks = [process_chunk(chunk, i, len(time_chunks), video_duration) for i, chunk in enumerate(time_chunks)]
                print(f"Starting parallel analysis of {len(chunk_tasks)} chunks...")
                chunk_analyses = await asyncio.gather(*chunk_tasks)

                # Combine analyses with simplified approach
                print(f"Combining analyses from {len(chunk_analyses)} chunks...")
                final_analysis = finalize_segments(chunk_analyses, video_duration)
                
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
                        "timestamps": [{"timestamp": 0, "description": "Processing error", "topic": "Error", "importance": 1}]
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
            model=os.getenv('MODEL_ID'),
            response_model=ChatResponse,
            messages=[{"role": "user", "content": prompt}],
            temperature=MODEL_TEMPERATURE
        )
        
        return {"answer": response.answer}
    except Exception as e:
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