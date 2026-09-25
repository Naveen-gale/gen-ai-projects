# ResearchMind - Multi-Agent AI Research System

A production-ready multi-agent AI pipeline that searches the web, scrapes content, writes a structured research report, and critiques it - all in one click.

## Live Demo
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

## Architecture

`
User Input (topic)
      |
  [Search Agent]  -- web_search tool (Tavily)
      |
  [Reader Agent]  -- scrape_url tool (BeautifulSoup)
      |
  [Writer Chain]  -- LLM generates structured report
      |
  [Critic Chain]  -- LLM scores and reviews the report
      |
  Final Report + Feedback (downloadable .md)
`

## Tech Stack
- **Frontend**: Streamlit
- **Agents**: LangChain + LangGraph
- **LLM**: Google Gemini 2.5 Flash (primary) / Mistral open-mistral-nemo (fallback)
- **Search**: Tavily API
- **Scraping**: BeautifulSoup4 + Requests

## Local Setup

### 1. Clone the repo
`ash
git clone https://github.com/your-username/researchmind.git
cd researchmind
`

### 2. Create virtual environment
`ash
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate  # Mac/Linux
`

### 3. Install dependencies
`ash
pip install -r requirements.txt
`

### 4. Set up API keys
`ash
cp .env.example .env
`
Edit .env and fill in your keys:
| Key | Where to get it | Cost |
|-----|----------------|------|
| GOOGLE_API_KEY | https://aistudio.google.com/app/apikey | **Free** |
| TAVILY_API_KEY | https://tavily.com | **Free** (1000 req/month) |
| MISTRALAI_API_KEY | https://console.mistral.ai | Free fallback |

### 5. Run
`ash
streamlit run app.py
`

## Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to https://share.streamlit.io
3. Click **New app** -> select your repo -> set main file: pp.py
4. Go to **Settings > Secrets** and add:
`	oml
GOOGLE_API_KEY = "your_key_here"
TAVILY_API_KEY = "your_key_here"
`
5. Click **Deploy**

## Project Structure
`
researchmind/
   app.py           # Streamlit UI (main entry point)
   agents.py        # LLM agents + chains
   tool.py          # web_search + scrape_url tools
   pipline.py       # CLI pipeline runner
   requirements.txt # Python dependencies
   .env.example     # Environment variable template
   .gitignore       # Git ignore rules
   .streamlit/
      secrets.toml  # Local secrets (NOT committed)
`

## License
MIT