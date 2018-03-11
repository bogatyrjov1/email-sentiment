from apiclient.discovery import build
import pandas as pd
import time
from conf import conf_sandbox

youtube = build(
    conf_sandbox["youtube"]["api_service_name"],
    conf_sandbox["youtube"]["api_version"],
    developerKey=conf_sandbox["youtube"]["developer_key"]
)


def get_videos(youtube, channelId, order):
    search_response = youtube.search().list(
        channelId=channelId,
        type="video",
        part="id,snippet",
        maxResults=50,
        order=order
    ).execute()

    return search_response.get("items", [])


def get_comment_threads():
    videos = get_videos(youtube, conf_sandbox["youtube"]["channel_id"], "viewCount")

    temp_comments = []
    for video in videos:
        time.sleep(1.0)
        results = youtube.commentThreads().list(
            part="snippet",
            videoId=video["id"]["videoId"],
            textFormat="plainText",
            maxResults=20,
            order='relevance'
        ).execute()

        for item in results["items"]:
            comment = item["snippet"]["topLevelComment"]
            temp_comments.append(dict(
                videoId=video["id"]["videoId"],
                videoName=video["snippet"]["title"],
                nbrReplies=item["snippet"]["totalReplyCount"],
                author=comment["snippet"]["authorDisplayName"],
                likes=comment["snippet"]["likeCount"],
                publishedAt=comment["snippet"]["publishedAt"],
                text=comment["snippet"]["textDisplay"].encode('utf-8').strip()
            ))

    return temp_comments


def get_video_infos(videos):
    video_list = {}
    for search_result in videos:
        if search_result["id"]["kind"] == "youtube#video":
            video_list[search_result["id"]["videoId"]] = search_result["snippet"]["title"]

    s = ','.join(video_list.keys())
    videos_list_response = youtube.videos().list(id=s, part='id,statistics').execute()
    res = []
    for i in videos_list_response['items']:
        temp_res = dict(v_title = video_list[i['id']])
        temp_res.update(i['statistics'])
        res.append(temp_res)

    data = pd.DataFrame.from_dict(res)
    data['viewCount'] = data['viewCount'].map(lambda x : float(x))
    data['commentCount'] = data['commentCount'].map(lambda x : float(x))

    return data


def get_videos_sorted():
    videos = get_videos(youtube, conf_sandbox["youtube"]["channel_id"], "viewCount")
    videos_data = get_video_infos(videos)

    return videos_data.sort_values(by=['viewCount'], ascending=0).head(20)
