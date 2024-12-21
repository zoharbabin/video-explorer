# Video Explorer

A sophisticated web application that uses Kaltura and AWS Bedrock (Claude 3) to analyze videos and enable interactive conversations with their content. The application provides deep video analysis, topic extraction, and AI-powered chat capabilities.

## Features

- **Video Management**
  - Search Kaltura videos by category or text
  - Browse recent videos with thumbnails and descriptions
  - Multi-video selection for batch analysis

- **AI Analysis**
  - Deep content analysis using AWS Bedrock (Claude 3)
  - Topic extraction and importance scoring
  - Key moment detection with timestamps
  - Comprehensive video summaries
  - Intelligent segmentation of long videos

- **Interactive Interface**
  - Clean, responsive design using PicoCSS
  - Tabbed analysis view (Summary, Insights, Topics, Key Moments)
  - Interactive topic visualization
  - Video segment preview with thumbnails
  - Real-time chat with video content
  - Dark theme support

## Prerequisites

- Python 3.9 or higher
- Kaltura account with API access
- AWS account with Bedrock access (Claude 3 model)
- Node.js and npm (for development)

## Quick Start

1. Clone the repository:
```bash
git clone [your-repo-url]
cd video-explorer
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up configuration:
```bash
cp .env.example .env
```

Edit `.env` with your credentials:
- Get Kaltura credentials from your Kaltura Management Console
- Get AWS credentials with Bedrock access from AWS Console

4. Run the application:
```bash
python main.py
```

5. Open http://localhost:8000 in your browser

## Configuration

The `.env` file supports the following configuration options:

```env
# Kaltura Configuration
KALTURA_PARTNER_ID=your_partner_id
KALTURA_SECRET=your_secret_key
KALTURA_SERVICE_URL=https://cdnapisec.kaltura.com
KALTURA_SESSION_DURATION=86400

# AWS Configuration
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1  # Region where Bedrock is available

# Server Configuration
HOST=0.0.0.0
PORT=8000

# Analysis Configuration
MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
MODEL_TIMEOUT=60
MODEL_MAX_TOKENS=4000
MODEL_CHUNK_SIZE=24000
MODEL_TEMPERATURE=0
PAGE_SIZE=10
```

## Architecture

### Backend
- **FastAPI**: Modern, fast web framework for building APIs
- **Kaltura API**: Video content management and delivery
- **AWS Bedrock**: AI analysis using Claude 3 model
- **litellm**: LLM integration layer
- **instructor**: Structured outputs from LLM responses

### Frontend
- **HTML/JavaScript**: Pure JavaScript for interactivity
- **PicoCSS**: Minimal, semantic CSS framework
- **Responsive Design**: Mobile-first approach
- **Dynamic UI**: Real-time updates and animations

### Key Features
1. Parallel processing of video analysis
2. Intelligent chunking for long videos
3. In-memory caching of analysis results
4. Real-time progress tracking
5. Structured AI responses for consistent output

## API Endpoints

- `GET /`: Main application interface
- `GET /api/videos`: Search and list videos
- `POST /api/analyze`: Analyze selected videos
- `GET /api/analysis-progress/{task_id}`: Check analysis progress
- `POST /api/chat`: Chat with analyzed video content

## Development

### Local Development
```bash
# Run with auto-reload
python main.py
```

### Code Structure
```
video-explorer/
├── main.py           # FastAPI application and backend logic
├── static/
│   └── style.css    # Application styling
├── templates/
│   └── index.html   # Frontend interface
└── requirements.txt  # Python dependencies
```

## Deployment

### Deploy to AWS Elastic Beanstalk

1. Install EB CLI:
```bash
pip install awsebcli
```

2. Initialize EB project:
```bash
eb init -p python-3.9 video-explorer
```

3. Create and deploy:
```bash
eb create video-explorer-env
```

### Deploy to Heroku

1. Install Heroku CLI and login:
```bash
heroku login
```

2. Create and deploy:
```bash
heroku create video-explorer
git push heroku main
```

3. Set environment variables:
```bash
heroku config:set KALTURA_PARTNER_ID=your_partner_id
heroku config:set KALTURA_SECRET=your_secret_key
heroku config:set AWS_ACCESS_KEY_ID=your_aws_access_key
heroku config:set AWS_SECRET_ACCESS_KEY=your_aws_secret_key
heroku config:set AWS_REGION=us-east-1
```

## Browser Support

- Chrome/Edge (latest 2 versions)
- Firefox (latest 2 versions)
- Safari (latest 2 versions)
- Mobile browsers (iOS Safari, Android Chrome)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.