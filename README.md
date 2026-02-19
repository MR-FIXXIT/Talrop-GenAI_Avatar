# Conversatoinal Avatar


## Features
- **Authentication**: Secure authentication mechanisms.
- **Document Management**: Upload, ingest, and manage documents.
- **Scraping**: Web scraping capabilities for data extraction.
- **Chat RAG**: Retrieval-augmented generation for chat systems.
- **API Endpoints**: Organized routes for managing organizations, documents, and keys.
- **Pinecone Integration**: Efficient vector database setup and management.

## Project Structure
```
.
├── auth.py                # Authentication-related utilities
├── db.py                  # Database connection and management
├── main.py                # Entry point of the application
├── pinecone_client.py     # Pinecone client setup
├── pinecone_setup.py      # Pinecone index setup
├── requirements.txt       # Python dependencies
├── models/                # Data models
├── rag/                   # Retrieval-augmented generation logic
├── routes/                # API routes
├── schemas/               # Data schemas
├── uploads/               # Uploaded files
├── utils/                 # Utility scripts
```

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```

2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up Pinecone:
   - Configure your Pinecone API key in `pinecone_client.py`.
   - Run `pinecone_setup.py` to initialize the index.

## Usage
1. Start the application:
   ```bash
   python main.py
   ```

2. Access the API endpoints:
   - Health Check: `/health`
   - Chat: `/chat`
   - Document Management: `/uploads`
   - Organization Management: `/orgs`
