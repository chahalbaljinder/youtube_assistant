#!/usr/bin/env python3
"""
Test script for YouTube Assistant
"""
import langchain_helper as lch

def test_youtube_assistant():
    """Test the YouTube Assistant functionality"""
    
    # Test video URL (make sure it has captions)
    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Rick Astley - Never Gonna Give You Up
    
    print("🎥 Testing YouTube Assistant...")
    print(f"🔗 Video URL: {video_url}")
    
    try:
        # Test creating database from YouTube video
        print("\n📊 Creating FAISS database from YouTube video...")
        db = lch.create_db_from_youtube_video_url(video_url)
        print("✅ Database created successfully!")
        
        # Test querying the database
        print("\n❓ Testing query functionality...")
        query = "What is this video about?"
        response, docs = lch.get_response_from_query(db, query)
        
        print(f"🤖 Query: {query}")
        print(f"📝 Response: {response}")
        print(f"📄 Number of relevant documents: {len(docs)}")
        
        print("\n🎉 Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_youtube_assistant()
    exit(0 if success else 1)
