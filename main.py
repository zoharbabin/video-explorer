import os
import json
import time
import asyncio
import logging
import base64
from typing import List, Optional, Dict
import requests
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from KalturaClient.exceptions import KalturaException
from pydantic import BaseModel, Field
import instructor
from dotenv import load_dotenv
from litellm import completion
from logger_config import setup_logging
from kaltura_utils import KalturaUtils

# Set up logging configuration
setup_logging(os.getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# In-memory storage
analysis_cache: Dict[str, dict] = {}
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

# Configuration values
KALTURA_SESSION_DURATION = int(os.getenv('KALTURA_SESSION_DURATION', '86400'))
PAGE_SIZE = int(os.getenv('PAGE_SIZE', '10'))
MODEL_ID = os.getenv('MODEL_ID', 'anthropic.claude-3-5-sonnet-20241022-v2:0')
SOCIAL_POST_MODEL_ID = os.getenv('SOCIAL_POST_MODEL_ID', 'anthropic.claude-3-5-haiku-20241022-v1:0')
MODEL_TIMEOUT = int(os.getenv('MODEL_TIMEOUT', '60'))
MODEL_MAX_TOKENS = int(os.getenv('MODEL_MAX_TOKENS', '4000'))
MODEL_CHUNK_SIZE = int(os.getenv('MODEL_CHUNK_SIZE', '24000'))
MODEL_TEMPERATURE = float(os.getenv('MODEL_TEMPERATURE', '0'))

# Initialize clients
kaltura = KalturaUtils(
    service_url=os.getenv('KALTURA_SERVICE_URL', 'https://cdnapisec.kaltura.com'),
    partner_id=int(os.getenv('KALTURA_PARTNER_ID')),
    admin_secret=os.getenv('KALTURA_SECRET'),
    session_duration=KALTURA_SESSION_DURATION
)

llm_client = instructor.from_litellm(completion)
os.environ["AWS_REGION"] = os.getenv('AWS_REGION', 'us-east-1')

# Pydantic models
class TimestampEntry(BaseModel):
    start_timestamp: float = Field(description="Timestamp in seconds")
    end_timestamp: float = Field(description="Timestamp in seconds")
    description: str = Field(description="Description of what occurs at this timestamp")
    topic: str = Field(description="Main topic being discussed at this timestamp")
    importance: int = Field(description="Importance level (1-5)", ge=1, le=5)
    thumbnails: Optional[List[str]] = Field(default=None, description="List of thumbnail URLs for this segment")
class SocialPost(BaseModel):
    linkedin_text: str = Field(description="LinkedIn post text (up to 3000 characters)")
    x_text: str = Field(description="X (Twitter) post text (up to 280 characters)")
    thumbnails: List[str] = Field(default_factory=list, description="List of thumbnail URLs to include")
    startTime: float = Field(description="Start timestamp of the clip in seconds")
    endTime: float = Field(description="End timestamp of the clip in seconds")

class VideoAnalysis(BaseModel):
    summary: str = Field(description="A comprehensive summary of the video content")
    insights: List[str] = Field(description="Key insights from the video")
    topics: List[dict] = Field(description="Main topics with importance scores")
    timestamps: List[TimestampEntry] = Field(description="Important timestamps")

class ChatResponse(BaseModel):
    answer: str = Field(description="Response based on video context")

class ChatRequest(BaseModel):
    question: str
    context: List[dict]

def init_kaltura_session() -> bool:
    """Initialize Kaltura session"""
    success, pid = kaltura.init_session()
    if not success:
        logger.error("Failed to initialize Kaltura session")
        return False
    logger.info("Successfully initialized Kaltura session for partner ID: %s", pid)
    return True

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/videos")
async def search_videos(category_id: Optional[str] = None, query: Optional[str] = None):
    """Search for videos with English captions"""
    try:
        if not init_kaltura_session():
            return {"error": "Failed to initialize Kaltura session"}
        
        videos = kaltura.fetch_videos(
            category_ids=category_id,
            free_text=query,
            number_of_videos=PAGE_SIZE
        )
        
        logger.info("Found %d videos with captions", len(videos))
        return {"videos": videos}
        
    except (KalturaException, ValueError) as e:
        logger.error("Error in search_videos: %s", str(e), exc_info=True)
        return {"error": str(e)}

async def process_transcript_segment(
    segment_chunks: List[dict],
    segment_index: int,
    total_segments: int,
    video_duration: float
) -> List[TimestampEntry]:
    """Process a segment of video transcript"""
    try:
        segment_start = round(segment_chunks[0][0]['startTime'] / 1000, 2)
        segment_end = round(segment_chunks[-1][-1]['endTime'] / 1000, 2)
        segment_length = segment_end - segment_start

        if segment_length < 15:
            logger.warning("Very short segment (only %.2f seconds)", segment_length)

        timestamped_lines = [
            f"[{round(entry['startTime']/1000, 2):.1f}s - {round(entry['endTime']/1000, 2):.1f}s] {entry['text']}"
            for chunk in segment_chunks
            for entry in chunk
        ]

        prompt = create_segment_analysis_prompt(
            segment_index,
            total_segments,
            segment_start,
            segment_end,
            timestamped_lines
        )

        segment_timestamps = await asyncio.to_thread(
            llm_client.chat.completions.create,
            model=MODEL_ID,
            response_model=List[TimestampEntry],
            messages=[{"role": "user", "content": prompt}],
            timeout=MODEL_TIMEOUT,
            max_tokens=MODEL_MAX_TOKENS,
            temperature=MODEL_TEMPERATURE
        )

        return validate_timestamps(
            segment_timestamps,
            segment_start,
            segment_end,
            video_duration
        )

    except (ValueError, TypeError) as e:
        logger.error("Error processing segment %d: %s", segment_index + 1, str(e), exc_info=True)
        return []

def create_segment_analysis_prompt(
    segment_index: int,
    total_segments: int,
    segment_start: float,
    segment_end: float,
    timestamped_lines: List[str]
) -> str:
    """Create prompt for segment analysis"""
    return f"""You have a video segment (Part {segment_index + 1} of {total_segments}), from {segment_start:.1f}s to {segment_end:.1f}s.

TRANSCRIPT (with exact timestamps for each line):
{chr(10).join(timestamped_lines)}

Instructions:
1. Identify key moments where primary discussed topics begin and end
2. Use exact timestamps from the transcript
3. Each key moment should have a clear description with verbatim words
4. Return JSON array of TimestampEntry objects
5. Ensure no overlapping timestamps
6. Cover entire segment without gaps
7. Each topic should be at least 30 seconds long

Example Output:
[
  {{
    "start_timestamp": 120.5,
    "end_timestamp": 234.0,
    "description": "Demonstrates neural network training interface",
    "topic": "Neural Network",
    "importance": 5
  }}
]"""

def validate_timestamps(
    timestamps: List[TimestampEntry],
    segment_start: float,
    segment_end: float,
    video_duration: float
) -> List[TimestampEntry]:
    """Validate and filter timestamps"""
    valid_results = []
    used_timestamps = set()

    for ts in timestamps:
        if (ts.start_timestamp < (segment_start - 0.1) or
            ts.end_timestamp > (segment_end + 0.1) or
            ts.end_timestamp > (video_duration + 0.1) or
            ts.start_timestamp >= ts.end_timestamp):
            continue

        timestamp_pair = (ts.start_timestamp, ts.end_timestamp)
        if timestamp_pair in used_timestamps:
            continue

        used_timestamps.add(timestamp_pair)
        valid_results.append(ts)

    return valid_results

async def generate_timestamps(transcript_chunks: List[dict], video_duration: float) -> List[TimestampEntry]:
    """Generate timestamps from transcript"""
    try:
        total_text_length = sum(len(entry['text']) for chunk in transcript_chunks for entry in chunk)
        chars_per_segment = 8000
        chunks_per_segment = max(1, int((chars_per_segment * len(transcript_chunks)) / total_text_length))
        
        segment_count = len(transcript_chunks) // chunks_per_segment + (1 if len(transcript_chunks) % chunks_per_segment else 0)
        
        segment_tasks = [
            process_transcript_segment(
                transcript_chunks[i:i + chunks_per_segment],
                i // chunks_per_segment,
                segment_count,
                video_duration
            )
            for i in range(0, len(transcript_chunks), chunks_per_segment)
        ]
        
        all_timestamps = [
            ts for result in await asyncio.gather(*segment_tasks)
            for ts in result
        ]
        
        return filter_timestamps(all_timestamps, video_duration)

    except (ValueError, TypeError) as e:
        logger.error("Error generating timestamps: %s", str(e), exc_info=True)
        return []

def filter_timestamps(timestamps: List[TimestampEntry], video_duration: float) -> List[TimestampEntry]:
    """Filter and validate timestamps"""
    filtered = []
    min_gap = 30

    for ts in sorted(timestamps, key=lambda x: x.start_timestamp):
        if ts.end_timestamp > video_duration:
            continue
            
        if not filtered or (ts.start_timestamp - filtered[-1].end_timestamp) >= min_gap:
            filtered.append(ts)

    target_count = int(video_duration / 180)
    if len(filtered) < target_count * 0.7:
        logger.warning("Generated only %d timestamps for %.1f minute video", len(filtered), video_duration / 60)

    return filtered

def finalize_segments(chunk_results: List[VideoAnalysis]) -> dict:
    """Combine chunk results into final analysis"""
    try:
        # Safely extract summaries and insights
        combined_summaries = [chunk.summary for chunk in chunk_results if chunk.summary]
        all_insights = [insight for chunk in chunk_results for insight in (chunk.insights or [])]
        
        # Safely process topics with validation
        topic_scores = {}
        for chunk in chunk_results:
            for topic in (chunk.topics or []):
                try:
                    # Handle different topic formats
                    if isinstance(topic, dict):
                        name = str(topic.get('name', topic.get('topic', '')))
                        score = int(topic.get('importance', topic.get('score', 3)))
                    else:
                        name = str(getattr(topic, 'name', getattr(topic, 'topic', str(topic))))
                        score = int(getattr(topic, 'importance', getattr(topic, 'score', 3)))
                    
                    if name:  # Only process if we have a valid name
                        topic_scores[name] = max(topic_scores.get(name, 0), score)
                except (AttributeError, ValueError, TypeError) as e:
                    logger.warning("Skipping invalid topic: %s. Error: %s", topic, str(e))
                    continue
        
        # Ensure we have at least one topic
        if not topic_scores:
            topic_scores["General"] = 3  # Default topic if none found
        
        return {
            "summary": "\n\n".join(combined_summaries) if combined_summaries else "No summary available.",
            "insights": list(dict.fromkeys(insight for insight in all_insights if insight)),
            "topics": [
                {"name": name, "importance": score}
                for name, score in sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
            ],
            "timestamps": []
        }
    except (KalturaException, ValueError, TypeError, RuntimeError) as e:
        logger.error("Error in finalize_segments: %s", str(e), exc_info=True)
        return {
            "summary": "Error processing analysis.",
            "insights": ["Error occurred while processing analysis"],
            "topics": [{"name": "Error", "importance": 1}],
            "timestamps": []
        }

@app.get("/api/analysis-progress/{task_id}")
async def get_analysis_progress(task_id: str):
    """Get video analysis task progress"""
    return (
        {"progress": analysis_progress[task_id]}
        if task_id in analysis_progress
        else {"error": "Task not found"}
    )

@app.post("/api/analyze")
async def analyze_videos(video_ids: List[str]):
    """Analyze videos using AI"""
    task_id = f"task_{len(video_ids)}_{int(time.time())}"
    analysis_progress[task_id] = 0
    results = []

    try:
        # Check cache
        videos_to_process = [
            vid for vid in video_ids
            if vid not in analysis_cache
        ]
        
        results.extend(analysis_cache[vid] for vid in video_ids if vid in analysis_cache)

        if not videos_to_process:
            return {"task_id": task_id, "results": results, "status": "completed"}

        if not init_kaltura_session():
            analysis_progress[task_id] = -1
            return {
                "error": "Failed to initialize Kaltura session",
                "task_id": task_id,
                "status": "failed"
            }

        video_results = await process_videos(videos_to_process, task_id, len(video_ids))
        results.extend(video_results)

        analysis_progress[task_id] = 100
        return {
            "task_id": task_id,
            "results": results,
            "status": "completed"
        }

    except (KalturaException, ValueError, TypeError, RuntimeError) as e:
        logger.error("Fatal error in analyze_videos: %s", str(e), exc_info=True)
        analysis_progress[task_id] = -1
        return {
            "error": str(e),
            "task_id": task_id,
            "status": "failed"
        }

async def process_videos(video_ids: List[str], task_id: str, total_videos: int) -> List[dict]:
    """Process multiple videos in parallel"""
    video_tasks = [
        process_video(video_id, task_id, total_videos)
        for video_id in video_ids
    ]
    return await asyncio.gather(*video_tasks)

async def process_video(video_id: str, task_id: str, total_videos: int) -> dict:
    """Process a single video"""
    try:
        captions = kaltura.get_english_captions(video_id)
        if not captions:
            return create_error_result(video_id, "No English captions found")

        transcript_chunks = kaltura.get_json_transcript(captions[0]['id'])
        if not transcript_chunks:
            return create_error_result(video_id, "No readable transcript content")

        video_info = kaltura.client.media.get(video_id) # pylint: disable=no-member
        video_duration = float(video_info.duration)

        chunk_analyses = await process_video_chunks(transcript_chunks)
        final_analysis = finalize_segments(chunk_analyses)

        timestamps = await generate_timestamps(transcript_chunks, video_duration)
        final_analysis["timestamps"] = timestamps

        result = {"video_id": video_id, "analysis": final_analysis}
        analysis_cache[video_id] = result
        analysis_progress[task_id] = min(((len(analysis_cache) + 1) / total_videos) * 100, 99)

        return result

    except (KalturaException, ValueError, TypeError) as e:
        logger.error("Error processing video %s: %s", video_id, str(e), exc_info=True)
        return create_error_result(video_id, str(e))

def create_error_result(video_id: str, error_message: str) -> dict:
    """Create standardized error result"""
    return {
        "video_id": video_id,
        "analysis": {
            "summary": "Error processing video.",
            "insights": [f"Error: {error_message}"],
            "topics": [{"name": "<ERROR>", "importance": 1}],
            "timestamps": [
                TimestampEntry(
                    start_timestamp=0,
                    end_timestamp=0,
                    description="Processing error",
                    topic="Error",
                    importance=1
                )
            ]
        }
    }

async def process_video_chunks(transcript_chunks: List[dict]) -> List[VideoAnalysis]:
    """Process video chunks in parallel"""
    chunk_tasks = []
    for i, chunk in enumerate(transcript_chunks):
        chunk_data = {
            "entries": chunk,
            "start": chunk[0]['startTime'] / 1000,
            "end": chunk[-1]['endTime'] / 1000
        }
        chunk_tasks.append(process_chunk(chunk_data, i, len(transcript_chunks)))

    return await asyncio.gather(*chunk_tasks)

async def process_chunk(chunk: dict, chunk_index: int, total_chunks: int) -> VideoAnalysis:
    """Process a single chunk"""
    try:
        timestamped_lines = [
            f"[{entry['startTime']/1000:.1f}s - {entry['endTime']/1000:.1f}s] {entry['text']}"
            for entry in chunk['entries']
        ]

        prompt = f"""Analyze this video transcript section ({chunk_index + 1} of {total_chunks}).

Transcript:
{chr(10).join(timestamped_lines)}

Instructions:
1. Provide a well-structured summary using markdown
2. Focus on core content and main ideas
3. Identify overarching themes and concepts

Format as VideoAnalysis object."""

        return await asyncio.to_thread(
            llm_client.chat.completions.create,
            model=MODEL_ID,
            response_model=VideoAnalysis,
            messages=[{"role": "user", "content": prompt}],
            timeout=MODEL_TIMEOUT,
            max_tokens=MODEL_MAX_TOKENS,
            temperature=MODEL_TEMPERATURE
        )

    except (ValueError, TypeError) as e:
        logger.error("Error processing chunk %d: %s", chunk_index, str(e), exc_info=True)
        return VideoAnalysis(
            summary=f"Error processing chunk {chunk_index}",
            insights=[f"Error: {str(e)}"],
            topics=[{"name": "<ERROR>", "importance": 1}],
            timestamps=[]
        )

@app.post("/api/generate-social-post/{video_id}")
async def generate_social_post(video_id: str, moment_id: int):
    """Generate a social media post for a specific key moment"""
    try:
        if not init_kaltura_session():
            return {"error": "Failed to initialize Kaltura session"}

        # Get video info and analysis from cache
        if video_id not in analysis_cache:
            return {"error": "Video analysis not found"}

        analysis = analysis_cache[video_id]["analysis"]
        if not analysis["timestamps"]:
            return {"error": "No key moments found"}

        # Get the specific moment
        if moment_id >= len(analysis["timestamps"]):
            return {"error": "Invalid moment ID"}
        
        moment = analysis["timestamps"][moment_id]
        
        def encode_image_from_url(url: str) -> str:
            """Download image from URL and encode as base64"""
            response = requests.get(url)
            response.raise_for_status()
            return base64.b64encode(response.content).decode("utf-8")

        try:
            # Generate thumbnail URL for middle of segment
            duration = moment.end_timestamp - moment.start_timestamp
            time_point = moment.start_timestamp + (duration / 2)  # Middle of segment
            
            # Get Kaltura session for thumbnail auth
            # ks = kaltura.client.getKs() # pylint: disable=no-member
            thumbnail_url = (
                f"https://cdnapi-ev.kaltura.com/p/{kaltura.partner_id}"
                f"/sp/{kaltura.partner_id}00/thumbnail/entry_id/{video_id}"
                f"/width/320/vid_sec/{time_point}" # /ks/{ks} - in most cases this is not really needed
            )

            # Create optimized message for LLM
            message = {
                "role": "user",
                "content": f"""Create two platform-optimized posts for this video moment.

MOMENT CONTEXT:
Topic: {moment.topic}
Description: {moment.description}
Duration: {duration:.1f} seconds

REQUIREMENTS:

LinkedIn Post (max 3000 chars):
- Professional and insightful tone
- Focus on business value and learning opportunities
- Include 2-3 relevant emojis
- End with thought-provoking question or clear call-to-action
- Keep hashtags minimal and professional

X Post (max 280 chars):
- Concise and attention-grabbing
- More casual and dynamic tone
- 1-2 impactful emojis
- Focus on key takeaway or surprising insight
- Include 2-3 relevant hashtags

Note: Focus solely on this specific moment's content, not the entire video.
Suggested hashtags: #ArtifactDevelopment #PythonInnovation #SoftwareEngineering #TechInnovation #DashboardSolutions

Return as SocialPost object with separate linkedin_text and x_text fields."""
            }

            # Generate post using LLM with timeout using faster model
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    llm_client.chat.completions.create,
                    model=SOCIAL_POST_MODEL_ID,
                    response_model=SocialPost,
                    messages=[message],
                    temperature=0.7,
                    max_tokens=1000  # Limit tokens for faster response
                ),
                timeout=15
            )

            # Generate default hashtags based on moment topic
            default_hashtags = [
                "#ArtifactDevelopment",
                "#PythonInnovation",
                "#SoftwareEngineering",
                "#TechInnovation",
                "#DashboardSolutions"
            ]

            # Add metadata to response
            response.thumbnails = [thumbnail_url]
            response.startTime = moment.start_timestamp
            response.endTime = moment.end_timestamp

            # Use default hashtags
            hashtags = ' '.join(default_hashtags)

            # Return platform-specific response format with consistent structure
            return {
                "linkedin": {
                    "text": response.linkedin_text,
                    "hashtags": hashtags
                },
                "x": {
                    "text": response.x_text,
                    "hashtags": hashtags
                },
                "metadata": {
                    "thumbnails": [thumbnail_url],
                    "startTime": moment.start_timestamp,
                    "endTime": moment.end_timestamp
                }
            }

        except asyncio.TimeoutError:
            logger.error("LLM request timed out while generating social post")
            return {
                "error": "Request timed out. Please try again.",
                "linkedin": {"text": "", "hashtags": ""},
                "x": {"text": "", "hashtags": ""},
                "metadata": {
                    "thumbnails": [thumbnail_url],
                    "startTime": moment.start_timestamp,
                    "endTime": moment.end_timestamp
                }
            }
        except (requests.RequestException, ValueError, TypeError) as e:
            error_msg = f"Failed to generate post: {str(e)}"
            logger.error("Error generating social post: %s", str(e), exc_info=True)
            return {
                "error": error_msg,
                "linkedin": {"text": "", "hashtags": ""},
                "x": {"text": "", "hashtags": ""},
                "metadata": {
                    "thumbnails": [thumbnail_url],
                    "startTime": moment.start_timestamp,
                    "endTime": moment.end_timestamp
                }
            }

    except (KalturaException, ValueError, TypeError, RuntimeError) as e:
        logger.error("Error generating social post: %s", str(e), exc_info=True)
        return {"error": str(e)}

@app.post("/api/chat")
async def chat_with_videos(request: ChatRequest):
    """Chat with analyzed videos"""
    try:
        prompt = f"""Question: {request.question}

Video Context:
{json.dumps(request.context, indent=2)}

Instructions:
Provide a clear, direct answer based on the video context above."""

        response = await asyncio.to_thread(
            llm_client.chat.completions.create,
            model=MODEL_ID,
            response_model=ChatResponse,
            messages=[{"role": "user", "content": prompt}],
            temperature=MODEL_TEMPERATURE
        )
        
        return {"answer": response.answer}

    except asyncio.TimeoutError:
        return {"error": "Request timed out. Please try again."}
    except (ValueError, TypeError) as e:
        logger.error("Chat error: %s", e, exc_info=True)
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv('HOST', '0.0.0.0'),
        port=int(os.getenv('PORT', '8000')),
        reload=True,
        server_header=False,
        proxy_headers=True
    )
