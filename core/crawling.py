from googleapiclient.discovery import build
from typing import List, Tuple, Optional
import logging
from config.settings import Settings

class YouTubeCrawler:
    def __init__(self):
        self.youtube = build("youtube", "v3", developerKey=Settings.YOUTUBE_API_KEY)
        self.logger = logging.getLogger(__name__)

    def get_video_details(self, video_id: str) -> Optional[Tuple[str, str]]:
        try:
            request = self.youtube.videos().list(part="snippet", id=video_id)
            response = request.execute()
            
            if not response["items"]:
                self.logger.warning(f"No video found for ID: {video_id}")
                return None

            video_info = response["items"][0]["snippet"]
            return (
                video_info["title"], 
                video_info["thumbnails"]["high"]["url"]
            )
        except Exception as e:
            self.logger.error(f"Error fetching video details: {e}")
            return None

    def get_comments(self, video_id: str, max_comments: int = 100) -> List[Tuple[str, str]]:
        comments = []
        try:
            request = self.youtube.commentThreads().list(
                part="snippet", 
                videoId=video_id, 
                maxResults=min(max_comments, 100)
            )

            while request and len(comments) < max_comments:
                response = request.execute()
                for item in response["items"]:
                    comment_snippet = item["snippet"]["topLevelComment"]["snippet"]
                    comments.append((
                        comment_snippet["textDisplay"], 
                        comment_snippet["authorDisplayName"]
                    ))
                
                request = self.youtube.commentThreads().list_next(request, response)
                if not request:
                    break

            return comments
        except Exception as e:
            self.logger.error(f"Error fetching comments: {e}")
            return []