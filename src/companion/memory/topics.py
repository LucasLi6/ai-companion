"""Canonical memory topics and deterministic identity queries.

A *topic* is the short, stable attribute a current-state memory is about
(``name``, ``location``, …). It lets consolidation recognise that two
statements describe the same attribute — and lets retrieval answer "what is my
name?" without the question sharing any wording with the stored fact.

Topics are advisory metadata attached by the model, so they are normalised and
constrained here on the trusted side. A model can never widen the canonical
set by inventing a topic that aliases a protected attribute.
"""

from __future__ import annotations

import re

# Canonical topic slugs. A small closed vocabulary keeps dedupe/correction and
# identity retrieval predictable; free-form topics are still accepted for other
# kinds (see ``canonicalize_topic``).
NAME = "name"
AGE = "age"
BIRTHDAY = "birthday"
LOCATION = "location"
OCCUPATION = "occupation"
EDUCATION = "education"
PREFERENCE = "preference"
RELATIONSHIP = "relationship"
PROJECT_STATE = "project_state"
GOAL = "goal"
HEALTH = "health"

CANONICAL_TOPICS = frozenset(
    {
        NAME,
        AGE,
        BIRTHDAY,
        LOCATION,
        OCCUPATION,
        EDUCATION,
        PREFERENCE,
        RELATIONSHIP,
        PROJECT_STATE,
        GOAL,
        HEALTH,
    }
)

_SYNONYMS: dict[str, str] = {
    # name
    "名字": NAME,
    "姓名": NAME,
    "名称": NAME,
    "称呼": NAME,
    "英文名": NAME,
    "name": NAME,
    "nickname": NAME,
    "full_name": NAME,
    "fullname": NAME,
    # age
    "年龄": AGE,
    "岁数": AGE,
    "年纪": AGE,
    "age": AGE,
    # birthday
    "生日": BIRTHDAY,
    "出生日期": BIRTHDAY,
    "出生": BIRTHDAY,
    "birthday": BIRTHDAY,
    "birth_date": BIRTHDAY,
    # location
    "住址": LOCATION,
    "地址": LOCATION,
    "城市": LOCATION,
    "所在地": LOCATION,
    "位置": LOCATION,
    "居住地": LOCATION,
    "location": LOCATION,
    "city": LOCATION,
    "address": LOCATION,
    "residence": LOCATION,
    # occupation
    "职业": OCCUPATION,
    "工作": OCCUPATION,
    "公司": OCCUPATION,
    "职位": OCCUPATION,
    "occupation": OCCUPATION,
    "job": OCCUPATION,
    "employer": OCCUPATION,
    # education
    "学校": EDUCATION,
    "学历": EDUCATION,
    "专业": EDUCATION,
    "education": EDUCATION,
    "school": EDUCATION,
    # preference
    "喜好": PREFERENCE,
    "爱好": PREFERENCE,
    "偏好": PREFERENCE,
    "兴趣": PREFERENCE,
    "preference": PREFERENCE,
    "likes": PREFERENCE,
    "hobby": PREFERENCE,
    # relationship
    "关系": RELATIONSHIP,
    "家人": RELATIONSHIP,
    "relationship": RELATIONSHIP,
    # project state
    "项目": PROJECT_STATE,
    "项目状态": PROJECT_STATE,
    "project": PROJECT_STATE,
    "project_state": PROJECT_STATE,
    "status": PROJECT_STATE,
    # goal
    "目标": GOAL,
    "计划": GOAL,
    "goal": GOAL,
    "plan": GOAL,
    # health
    "健康": HEALTH,
    "疾病": HEALTH,
    "过敏": HEALTH,
    "health": HEALTH,
    "allergy": HEALTH,
}

# A topic longer than this is not a short attribute slug; treat it as
# unusable rather than letting free prose into the dedupe key.
_MAX_TOPIC_LEN = 64
_TOPIC_CHARS = re.compile(r"[^0-9a-zA-Z_.:\-一-鿿]+")

_NAME_CONTENT = re.compile(
    r"(?:我的?名字(?:是|叫|为)|我叫|称呼我(?:为|叫)?)\s*([^\s，。！？,.!?；;]{1,32})"
)
_LOCATION_CONTENT = re.compile(
    r"我(?:现在|目前|已经)?(?:住在|居住在|来自|搬到了|搬到)\s*([^\s，。！？,.!?；;]{1,32})"
)
_AGE_CONTENT = re.compile(r"我(?:今年|现在)?\s*(\d{1,3})\s*岁")
_OCCUPATION_CONTENT = re.compile(
    r"我(?:是|是一名|是个|做)\s*([^\s，。！？,.!?；;]{1,24}"
    r"(?:工程师|老师|学生|医生|设计师|程序员|经理|律师|会计|研究员))"
)
_HEALTH_CONTENT = re.compile(r"我(?:对[^\s，。！？,.!?；;]{1,16})?过敏")
_PROJECT_CONTENT = re.compile(
    r"我(?:正在|正|在|目前|现在|最近|一直)?"
    r"(?:开发|研发|研究|编写|构建|搭建|制作|做|设计)"
    r"[^\s，。！？,.!?；;]{1,32}"
)


def canonicalize_topic(topic: str | None, content: str | None = None) -> str | None:
    """Return a normalised topic slug, inferring one only for clear cases.

    ``topic`` wins when usable. Otherwise a handful of unambiguous
    self-descriptions ("我叫…", "我住在…", "我今年…岁") supply a topic so
    identity facts remain findable even if the model omits metadata.
    """
    if topic:
        raw = topic.strip().strip("#").strip()
        lowered = raw.casefold()
        if lowered in _SYNONYMS:
            return _SYNONYMS[lowered]
        slug = _TOPIC_CHARS.sub("_", lowered).strip("_")
        if slug and len(slug) <= _MAX_TOPIC_LEN:
            return slug
    if not content:
        return None
    if _NAME_CONTENT.search(content):
        return NAME
    if _LOCATION_CONTENT.search(content):
        return LOCATION
    if _AGE_CONTENT.search(content):
        return AGE
    if _OCCUPATION_CONTENT.search(content):
        return OCCUPATION
    if _PROJECT_CONTENT.search(content):
        return PROJECT_STATE
    if _HEALTH_CONTENT.search(content):
        return HEALTH
    return None


_IDENTITY_QUERIES: list[tuple[str, re.Pattern[str]]] = [
    (
        NAME,
        re.compile(
            r"(?:我|你)(?:的)?(?:名字|姓名)(?:是|叫|叫什么|叫啥|是什么)"
            r"|我叫(?:什么|啥|作什么)|我是谁"
            r"|\b(?:what(?:'s| is) my name|do you (?:know|remember) my name"
            r"|who am i)\b",
            re.IGNORECASE,
        ),
    ),
    (
        LOCATION,
        re.compile(
            r"我(?:现在|目前)?(?:住在|住在哪|来自哪|在哪个城市|的?(?:住址|地址)是什么)"
            r"|\bwhere (?:do i live|am i from|do i come from)\b",
            re.IGNORECASE,
        ),
    ),
    (
        AGE,
        re.compile(
            r"我(?:今年)?(?:多大|几岁|多少岁)"
            r"|\bhow old am i\b",
            re.IGNORECASE,
        ),
    ),
    (
        BIRTHDAY,
        re.compile(
            r"我的?生日(?:是)?(?:什么时候|哪一天|哪天|几号|是什么)"
            r"|\bwhen is my birthday\b",
            re.IGNORECASE,
        ),
    ),
    (
        OCCUPATION,
        re.compile(
            r"我(?:是)?(?:做什么|干什么|做什么工作|什么职业)(?:的|工作)?"
            r"|\bwhat do i do(?: for a living)?\b|\bwhat(?:'s| is) my (?:job|occupation)\b",
            re.IGNORECASE,
        ),
    ),
    (
        PREFERENCE,
        re.compile(
            r"我(?:最|比较)?(?:喜欢|爱|偏好)(?:吃|喝|什么|做什么)"
            r"|\bwhat do i (?:like|prefer)\b",
            re.IGNORECASE,
        ),
    ),
    (
        GOAL,
        re.compile(
            r"我的?(?:目标|计划)(?:是)?(?:什么|啥)"
            r"|\bwhat(?:'s| is) my (?:goal|plan)\b",
            re.IGNORECASE,
        ),
    ),
    (
        HEALTH,
        re.compile(
            r"我对什么过敏|我(?:有什么)?(?:疾病|病史|过敏)"
            r"|\bdo i have (?:any )?allerg(?:y|ies)\b",
            re.IGNORECASE,
        ),
    ),
    (
        PROJECT_STATE,
        re.compile(
            r"我的?项目(?:现在|目前)?(?:进展|状态|怎么样)"
            r"|\bwhat(?:'s| is) the (?:status|state) of my project\b",
            re.IGNORECASE,
        ),
    ),
]


def detect_identity_query(text: str | None) -> list[str]:
    """Return canonical topics an identity question is asking about.

    Ordered most-specific first and de-duplicated. Empty means "not an
    identity question"; the caller then falls back to keyword retrieval.
    """
    if not text:
        return []
    topics: list[str] = []
    for topic, pattern in _IDENTITY_QUERIES:
        if pattern.search(text) and topic not in topics:
            topics.append(topic)
    return topics
