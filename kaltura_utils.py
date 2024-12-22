import json
import time
import logging
from typing import List, Optional, Dict, Tuple
import requests
from KalturaClient import KalturaClient, KalturaConfiguration
from KalturaClient.Base import IKalturaLogger
from KalturaClient.exceptions import KalturaClientException, KalturaException
from KalturaClient.Plugins.Core import (
    KalturaBaseEntryFilter,
    KalturaFilterPager,
    KalturaMediaType,
    KalturaSessionType
)
from KalturaClient.Plugins.Caption import (
    KalturaCaptionAssetFilter,
    KalturaCaptionAssetOrderBy,
    KalturaLanguage
)
from KalturaClient.Plugins.ElasticSearch import (
    KalturaESearchEntryParams,
    KalturaESearchEntryOperator,
    KalturaESearchOperatorType,
    KalturaESearchCaptionItem,
    KalturaESearchCaptionFieldName,
    KalturaESearchItemType,
    KalturaESearchEntryItem,
    KalturaESearchEntryFieldName,
    KalturaESearchOrderBy,
    KalturaESearchEntryOrderByItem,
    KalturaESearchEntryOrderByFieldName,
    KalturaESearchSortOrder,
    KalturaESearchCategoryEntryItem,
    KalturaESearchCategoryEntryFieldName,
    KalturaCategoryEntryStatus,
    KalturaESearchUnifiedItem
)

# Set up logging for this module
logger = logging.getLogger(__name__)

class KalturaLogger(IKalturaLogger):
    def __init__(self):
        self.logger = logging.getLogger('KalturaClient')
        
    def log(self, msg):
        self.logger.debug(msg)

class CustomKalturaClient(KalturaClient):
    def __init__(self, config, max_retries=3, delay=1, backoff=2):
        super().__init__(config)
        self.max_retries = max_retries
        self.delay = delay
        self.backoff = backoff

    def retry_on_exception(self, func, *args, **kwargs):
        mtries, mdelay = self.max_retries, self.delay
        while mtries > 1:
            try:
                return func(*args, **kwargs)
            except (KalturaException, KalturaClientException) as error:
                if self._is_non_retryable_error(error):
                    raise
                msg = f"{str(error)}, Kaltura API retrying request in {mdelay} seconds..."
                context = f'Function "{func.__name__}" failed on attempt {self.max_retries - mtries + 1}'
                logger.warning('Retrying function due to error: %s Context: %s', msg, context)
                time.sleep(mdelay)
                mtries -= 1
                mdelay *= self.backoff
        return func(*args, **kwargs)

    def _is_non_retryable_error(self, error):
        non_retryable_errors = ["INVALID_KS"]
        if isinstance(error, (KalturaException, KalturaClientException)):
            error_code = getattr(error, 'code', None)
            return error_code in non_retryable_errors
        return False

class KalturaUtils:
    def __init__(self, service_url: str, partner_id: int, admin_secret: str, session_duration: int = 86400):
        self.service_url = service_url
        self.partner_id = partner_id
        self.admin_secret = admin_secret
        self.session_duration = session_duration
        
        # Initialize Kaltura configuration
        config = KalturaConfiguration()
        config.serviceUrl = self.service_url
        config.setLogger(KalturaLogger())
        
        # Initialize custom client with retries
        self.client = CustomKalturaClient(config)
        
    def init_session(self) -> Tuple[bool, int]:
        """Initialize Kaltura session and return (success, partner_id)"""
        try:
            # pylint: disable=no-member
            session = self.client.session.start(
                self.admin_secret,
                None,
                KalturaSessionType.ADMIN,
                self.partner_id,
                self.session_duration,
                "appid:video-explorer"
            )
            self.client.setKs(session) # pylint: disable=no-member
            
            # Verify session with test API call
            test_filter = KalturaBaseEntryFilter()
            test_pager = KalturaFilterPager()
            test_pager.pageSize = 1
            self.client.baseEntry.list(test_filter, test_pager) # pylint: disable=no-member
            
            return True, self.partner_id
            
        except (KalturaClientException, KalturaException) as e:
            logger.error("Failed to initialize Kaltura session: %s", e)
            return False, -1

    def fetch_videos(self, category_ids: Optional[str] = None, free_text: Optional[str] = None, number_of_videos: int = 6) -> List[Dict]:
        """Fetch videos with captions using ElasticSearch plugin"""
        search_params = KalturaESearchEntryParams()
        search_params.orderBy = KalturaESearchOrderBy()
        order_item = KalturaESearchEntryOrderByItem()
        order_item.sortField = KalturaESearchEntryOrderByFieldName.CREATED_AT
        order_item.sortOrder = KalturaESearchSortOrder.ORDER_BY_DESC
        search_params.orderBy.orderItems = [order_item]

        search_params.searchOperator = KalturaESearchEntryOperator()
        search_params.searchOperator.operator = KalturaESearchOperatorType.AND_OP
        search_params.searchOperator.searchItems = []

        # Only fetch videos with captions
        caption_item = KalturaESearchCaptionItem()
        caption_item.fieldName = KalturaESearchCaptionFieldName.CONTENT
        caption_item.itemType = KalturaESearchItemType.EXISTS
        search_params.searchOperator.searchItems.append(caption_item)

        # Only fetch video type entries
        entry_item = KalturaESearchEntryItem()
        entry_item.fieldName = KalturaESearchEntryFieldName.MEDIA_TYPE
        entry_item.addHighlight = False
        entry_item.itemType = KalturaESearchItemType.EXACT_MATCH
        entry_item.searchTerm = str(KalturaMediaType.VIDEO)
        search_params.searchOperator.searchItems.append(entry_item)

        if category_ids:
            category_item = KalturaESearchCategoryEntryItem()
            category_item.categoryEntryStatus = KalturaCategoryEntryStatus.ACTIVE
            category_item.fieldName = KalturaESearchCategoryEntryFieldName.ANCESTOR_ID
            category_item.addHighlight = False
            category_item.itemType = KalturaESearchItemType.EXACT_MATCH
            category_item.searchTerm = category_ids
            search_params.searchOperator.searchItems.append(category_item)

        if free_text:
            unified_item = KalturaESearchUnifiedItem()
            unified_item.searchTerm = free_text
            unified_item.itemType = KalturaESearchItemType.PARTIAL
            search_params.searchOperator.searchItems.append(unified_item)

        pager = KalturaFilterPager()
        pager.pageSize = number_of_videos
        pager.pageIndex = 1

        result = self.client.elasticSearch.eSearch.searchEntry(search_params, pager) # pylint: disable=no-member

        videos = []
        for entry in result.objects:
            video_info = {
                "id": entry.object.id,
                "name": entry.object.name,
                "description": entry.object.description or "",
                "duration": entry.object.msDuration / 1000 if entry.object.msDuration else 0,
                "thumbnail_url": entry.object.thumbnailUrl,
                "created_at": entry.object.createdAt,
                "media_type": entry.object.mediaType.value,
                "tags": entry.object.tags or ""
            }
            videos.append(video_info)

        return videos

    def get_english_captions(self, entry_id: str) -> List[Dict]:
        """Get English captions for a video entry"""
        caption_filter = KalturaCaptionAssetFilter()
        caption_filter.entryIdEqual = entry_id
        caption_filter.languageEqual = KalturaLanguage.EN
        caption_filter.orderBy = KalturaCaptionAssetOrderBy.CREATED_AT_DESC
        pager = KalturaFilterPager()
        
        result = self.client.caption.captionAsset.list(caption_filter, pager) # pylint: disable=no-member
        return [{'id': caption.id, 'label': caption.label, 'language': caption.language} 
                for caption in result.objects]

    def get_json_transcript(self, caption_asset_id: str) -> List[Dict]:
        """
        Fetch and chunk a JSON transcript.
        """
        try:
            # Retrieve the URL for the JSON transcript
            cap_json_url = self.client.caption.captionAsset.serveAsJson(caption_asset_id) # pylint: disable=no-member
            timeout = (10, 30)  # Timeout: 10 seconds to connect, 30 seconds to read
            response = requests.get(cap_json_url, timeout=timeout)
            response.raise_for_status()

            # Parse the JSON response
            json_data = response.json()

            # Extract and validate transcript data
            transcript = json_data.get('objects', [])
            if not isinstance(transcript, list):
                logger.error("Unexpected data format for transcript: %s", type(transcript))
                return []

            return self.chunk_transcript(transcript) if transcript else []

        except requests.Timeout:
            logger.error("Request to %s timed out.", cap_json_url)
            return []
        except requests.ConnectionError:
            logger.error("Connection error occurred while accessing %s.", cap_json_url)
            return []
        except requests.RequestException as req_error:
            logger.error("Request failed: %s", req_error)
            return []
        except json.JSONDecodeError as json_error:
            logger.error("Error decoding JSON response from %s: %s", cap_json_url, json_error)
            return []
        except Exception as unexpected_error:  # Catch unexpected exceptions explicitly
            logger.error("An unexpected error occurred: %s", unexpected_error)
            raise  # Re-raise the exception after logging it

    def chunk_transcript(self, data: List[Dict], max_chars: int = 150000, overlap: int = 10000) -> List[Dict]:
        """Chunk transcript into segments with overlap"""
        def get_json_size(segment):
            return len(json.dumps(segment))

        # Order transcript by startTime
        data.sort(key=lambda x: x['startTime'])
        
        segments = []
        current_segment = []
        text_buffer = ''
        
        for entry in data:
            for content in entry['content']:
                sentences = content['text'].split('\n')
                for sentence in sentences:
                    if sentence:
                        sentence += '\n'
                        if not current_segment:
                            current_segment.append({
                                'startTime': entry['startTime'],
                                'endTime': entry['endTime'],
                                'text': sentence.strip()
                            })
                        else:
                            temp_segment = current_segment + [{
                                'startTime': entry['startTime'],
                                'endTime': entry['endTime'],
                                'text': sentence.strip()
                            }]
                            temp_size = get_json_size(temp_segment)
                            
                            if temp_size > max_chars:
                                segments.append(current_segment)
                                overlap_text = text_buffer[-overlap:].strip()
                                current_segment = [{
                                    'startTime': entry['startTime'],
                                    'endTime': entry['endTime'],
                                    'text': overlap_text
                                }]
                            else:
                                current_segment.append({
                                    'startTime': entry['startTime'],
                                    'endTime': entry['endTime'],
                                    'text': sentence.strip()
                                })
                            
                            text_buffer += sentence
        
        if current_segment:
            segments.append(current_segment)
            
        return segments