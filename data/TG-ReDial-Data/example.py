import pickle
import json

with open("./datasets/TG-ReDia-Data/train_data.pkl", "rb") as f:
    train_data = pickle.load(f)

print(len(train_data))  # 8495
print(type(train_data[0]))  # dict
print(train_data[0].keys())
key_list = list(
    train_data[0].keys()
)  # dict_keys(['conv_id', 'messages', 'goal_path', 'mentionMovies', 'user_id'])

# print(json.dumps(train_data[0], indent=2, ensure_ascii=False))
"""
{
  "conv_id": 0,
  "messages": [
    {
      "local_id": 1,
      "role": "Recommender",
      "content": "最近怎么样"
    },
    {
      "local_id": 2,
      "role": "Seeker",
      "content": "最近还可以，还是一个人，在上海奋斗，每天打卡，再无其他。你怎么样？"
    },
    {
      "local_id": 3,
      "role": "Recommender",
      "content": "我最近还不错啦，土豪金下周到手，是公司奖励。"
    },
    {
      "local_id": 4,
      "role": "Seeker",
      "content": "真是优秀啊！我不太在意奖励的，奖励就先不提了，还是脚踏实地认真工作吧，做个努力上进的好孩子吧。"
    },
    {
      "local_id": 5,
      "role": "Recommender",
      "content": "确实是，工作态度才是最重要的。这让我回忆起很多刚步入社会时候的事情。"
    },
    {
      "local_id": 6,
      "role": "Seeker",
      "content": "说到这，有没有值得回忆的电影推荐一下？"
    },
    {
      "local_id": 7,
      "role": "Recommender",
      "content": "《精灵鼠小弟》怎么样，经典的童年回忆片啊，用勇气和智慧来的一场冒险之旅。"
    },
    {
      "local_id": 8,
      "role": "Seeker",
      "content": "这个电影确实很好看，原来你的童年是喜欢小老鼠的呀。我小时候只觉得老鼠太狡猾，总是觉得老虎多可爱呢。果然每个人的童年是不一样的，哈哈。"
    },
    {
      "local_id": 9,
      "role": "Recommender",
      "content": "我童年时候不止喜欢小老鼠，还有其他的喜欢的动画，我推荐给你看看。"
    },
    {
      "local_id": 10,
      "role": "Seeker",
      "content": "好啊，越大反而越喜欢看小时候的童年影片，推荐一个满满童年回忆的影片吧！"
    },
    {
      "local_id": 11,
      "role": "Recommender",
      "content": "你会喜欢《机智的山羊》的，小时候就觉得山羊不可能这么聪明，但是这个影片让我觉得山羊还是很机智的。"
    },
    {
      "local_id": 12,
      "role": "Seeker",
      "content": "嗯，这个影片确实很不错，如果没看过这部影片只能说明你没有童年了。现在的小娃娃们也一定非常喜欢这部电影。"
    },
    {
      "local_id": 13,
      "role": "Recommender",
      "content": "是啊，好看的经典动画是充满了满满的回忆。"
    },
    {
      "local_id": 14,
      "role": "Seeker",
      "content": "你的童年里还有什么值得回忆的好影片吗？"
    },
    {
      "local_id": 15,
      "role": "Recommender",
      "content": "我推荐个《我知道》，这个是让我看过好几遍都不腻的动画片，是部有教育意义的动画片。"
    },
    {
      "local_id": 16,
      "role": "Seeker",
      "content": "恩，确实是典型的国产动画，这些充满童年回忆的动画，我感觉都很棒，谢谢啦，再见！"
    }
  ],
  "goal_path": {
    "2": [
      "Seeker",
      "谈论",
      "奋斗"
    ],
    "3": [
      "Rec",
      "谈论",
      "奖励"
    ],
    "4": [
      "Seeker",
      "拒绝",
      "奖励",
      "谈论",
      "孩子"
    ],
    "5": [
      "Rec",
      "谈论",
      "回忆"
    ],
    "6": [
      "Seeker",
      "请求推荐",
      [
        "回忆"
      ]
    ],
    "7": [
      "Rec",
      "推荐电影",
      "精灵鼠小弟"
    ],
    "8": [
      "Seeker",
      "反馈",
      null,
      "谈论",
      "童年"
    ],
    "9": [
      "Rec",
      "请求推荐",
      "童年"
    ],
    "10": [
      "Seeker",
      "允许推荐",
      [
        "回忆",
        "童年"
      ]
    ],
    "11": [
      "Rec",
      "推荐电影",
      "机智的山羊"
    ],
    "12": [
      "Seeker",
      "反馈",
      null,
      "谈论",
      "娃娃"
    ],
    "13": [
      "Rec",
      "谈论",
      "回忆"
    ],
    "14": [
      "Seeker",
      "请求推荐",
      [
        "回忆"
      ]
    ],
    "15": [
      "Rec",
      "推荐电影",
      "我知道"
    ],
    "16": [
      "Seeker",
      "反馈，结束",
      null
    ]
  },
  "mentionMovies": {
    "7": [
      "1295242",
      "精灵鼠小弟(1999)"
    ],
    "11": [
      "2123620",
      "机智的山羊(0)"
    ],
    "15": [
      "2252726",
      "我知道(1956)"
    ]
  },
  "user_id": "0"
}
"""
