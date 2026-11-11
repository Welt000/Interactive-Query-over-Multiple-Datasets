from __future__ import annotations

import csv
import hashlib
import math
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = (
    BASE_DIR
    / "original_datasets"
    / "Used_Car_4d_userstudy_1000x10"
)
DEFAULT_RESULT_DIR = Path(__file__).resolve().parent / "results"

METHODS = [
    "SUR-HDPI",
    "SUR-RH",
    "SUR-UH-Simplex",
    "SUR-UH-Random",
    "SharedQ",
]

ATTRIBUTE_NAMES = [
    "价格优势",
    "年份优势",
    "里程优势",
    "车况",
]

AMAZON_RAW_CSV = BASE_DIR / "original_datasets" / "amz_uk_processed_data.csv"
VEHICLE_RAW_CSV = BASE_DIR / "original_datasets" / "vehicles.csv"

CATEGORY_ZH = {
    "Sports & Outdoors": "运动与户外",
    "Skin Care": "护肤",
    "Make-up": "彩妆",
    "Men": "男装",
    "Women": "女装",
    "Fragrances": "香水",
    "Manicure & Pedicure Products": "美甲与足部护理",
    "Luggage and travel gear": "箱包与旅行用品",
    "Bath & Body": "沐浴与身体护理",
    "Hair Care": "头发护理",
}

TITLE_PHRASE_ZH = {
    "sports outdoors": "运动户外",
    "skin care": "护肤",
    "make up": "彩妆",
    "make-up": "彩妆",
    "bath body": "沐浴身体护理",
    "hair care": "头发护理",
    "yoga leggings": "瑜伽打底裤",
    "running shoes": "跑步鞋",
    "walking shoes": "步行鞋",
    "sports bra": "运动内衣",
    "water bottle": "水杯",
    "travel bag": "旅行包",
    "makeup brush": "化妆刷",
    "face cream": "面霜",
    "body wash": "沐浴露",
    "shower gel": "沐浴露",
    "hair dryer": "吹风机",
    "nail polish": "指甲油",
    "cotton solid": "棉质纯色",
}

TITLE_WORD_ZH = {
    "premium": "高级",
    "inner": "内侧",
    "pocket": "口袋",
    "pockets": "口袋",
    "yoga": "瑜伽",
    "legging": "打底裤",
    "leggings": "打底裤",
    "cotton": "棉",
    "solid": "纯色",
    "women": "女士",
    "womens": "女士",
    "men": "男士",
    "mens": "男士",
    "girls": "女孩",
    "boys": "男孩",
    "unisex": "男女通用",
    "shirt": "衬衫",
    "shirts": "衬衫",
    "tshirt": "T恤",
    "tshirts": "T恤",
    "jacket": "夹克",
    "coat": "外套",
    "pants": "裤子",
    "shorts": "短裤",
    "shoes": "鞋",
    "trainers": "运动鞋",
    "sneakers": "运动鞋",
    "running": "跑步",
    "walking": "步行",
    "outdoor": "户外",
    "sports": "运动",
    "fitness": "健身",
    "gym": "健身",
    "bag": "包",
    "bags": "包",
    "luggage": "行李箱",
    "travel": "旅行",
    "backpack": "双肩包",
    "waterproof": "防水",
    "cream": "霜",
    "serum": "精华",
    "lotion": "乳液",
    "cleanser": "洁面",
    "moisturiser": "保湿霜",
    "moisturizer": "保湿霜",
    "face": "面部",
    "body": "身体",
    "bath": "沐浴",
    "shower": "淋浴",
    "gel": "凝胶",
    "soap": "香皂",
    "shampoo": "洗发水",
    "conditioner": "护发素",
    "hair": "头发",
    "dryer": "吹风机",
    "brush": "刷",
    "comb": "梳子",
    "makeup": "彩妆",
    "foundation": "粉底",
    "lipstick": "口红",
    "mascara": "睫毛膏",
    "eyeliner": "眼线笔",
    "nail": "指甲",
    "polish": "油",
    "fragrance": "香水",
    "perfume": "香水",
    "deodorant": "除臭剂",
    "set": "套装",
    "pack": "装",
    "pcs": "件",
    "piece": "件",
    "pieces": "件",
    "with": "带",
    "for": "适用于",
    "and": "和",
    "black": "黑色",
    "white": "白色",
    "blue": "蓝色",
    "red": "红色",
    "green": "绿色",
    "pink": "粉色",
    "grey": "灰色",
    "gray": "灰色",
    "large": "大号",
    "small": "小号",
    "mini": "迷你",
}


def _split_numbers(text: str) -> list[float]:
    return [float(item) for item in text.replace(",", " ").split()]


def load_dataset(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(_split_numbers(line))
    if not rows:
        raise ValueError(f"empty dataset: {path}")

    first = rows[0]
    if len(first) == 2 and all(abs(x - round(x)) < 1e-9 for x in first):
        n = int(round(first[0]))
        d = int(round(first[1]))
        has_time_row = (
            len(rows) >= 2
            and len(rows[1]) in {1, 2}
            and len(rows) - 2 == n
            and all(len(row) == d for row in rows[2:])
        )
        start = 2 if has_time_row else 1
        body = rows[start:]
        if len(body) == n and all(len(row) == d for row in body):
            return np.asarray(body, dtype=float)
    return np.asarray(rows, dtype=float)


def dataset_label(path: Path) -> str:
    stem = path.stem
    if "_" in stem:
        stem = stem.split("_", 1)[1]
    return stem.replace("_", " ").title()


def translate_category(category: str) -> str:
    return CATEGORY_ZH.get(category, category)


def translate_title(title: str) -> str:
    text = " " + str(title).strip() + " "
    if not text.strip():
        return "未命名商品"
    for phrase, translated in sorted(TITLE_PHRASE_ZH.items(), key=lambda item: len(item[0]), reverse=True):
        text = re.sub(rf"\b{re.escape(phrase)}\b", f" {translated} ", text, flags=re.IGNORECASE)

    parts = re.split(r"([A-Za-z]+|\d+(?:\.\d+)?)", text)
    output: list[str] = []
    for part in parts:
        if not part:
            continue
        if re.fullmatch(r"[A-Za-z]+", part):
            output.append(TITLE_WORD_ZH.get(part.lower(), part))
        else:
            output.append(part)
    translated = re.sub(r"\s+", " ", "".join(output)).strip(" -|,;/")
    return translated or title


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            existing_header = next(reader, [])
        if existing_header and existing_header != fieldnames:
            backup_path = path.with_name(f"{path.stem}_legacy_{int(time.time())}{path.suffix}")
            path.replace(backup_path)
            exists = False
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


@dataclass
class AmazonStudyData:
    dataset_dir: Path = DEFAULT_DATASET_DIR
    paths: list[Path] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    datasets: list[np.ndarray] = field(default_factory=list)
    display_values: list[dict[tuple[float, ...], dict[str, Any]]] = field(default_factory=list)
    dim: int = 0

    @classmethod
    def load(cls, dataset_dir: Path = DEFAULT_DATASET_DIR) -> "AmazonStudyData":
        paths = sorted(
            path
            for path in dataset_dir.glob("*.txt")
            if path.name.lower() not in {"readme.txt", "metadata.txt"}
        )
        if not paths:
            raise FileNotFoundError(f"no txt datasets found in {dataset_dir}")
        datasets = [load_dataset(path) for path in paths]
        dim = int(datasets[0].shape[1])
        if any(data.shape[1] != dim for data in datasets):
            raise ValueError("all datasets must have the same dimension")
        if is_vehicle_dataset_dir(dataset_dir):
            labels = vehicle_category_labels(paths)
            display_values = build_vehicle_display_values(datasets)
        else:
            labels = amazon_category_labels(dataset_dir, paths) if dim == 4 else [dataset_label(path) for path in paths]
            display_values = build_amazon_display_values(dataset_dir, labels) if dim == 4 else [{} for _ in datasets]
        return cls(
            dataset_dir=dataset_dir,
            paths=paths,
            labels=labels,
            datasets=datasets,
            display_values=display_values,
            dim=dim,
        )


def valid_amazon_row(row: dict[str, str]) -> tuple[str, tuple[float, float, float, float]] | None:
    category = str(row.get("categoryName", "")).strip()
    stars = parse_raw_float(row.get("stars", ""))
    reviews = parse_raw_float(row.get("reviews", ""))
    price = parse_raw_float(row.get("price", ""))
    bought = parse_raw_float(row.get("boughtInLastMonth", "")) or 0.0
    if not category or stars is None or reviews is None or price is None:
        return None
    if stars <= 0 or reviews <= 0 or price <= 0 or bought < 0:
        return None
    return category, (price, stars, reviews, bought)


def amazon_category_labels(dataset_dir: Path, paths: list[Path]) -> list[str]:
    summary_path = dataset_dir / "dataset_summary.csv"
    if not summary_path.exists():
        return [dataset_label(path) for path in paths]
    labels: list[str] = []
    with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            category = str(row.get("category", "")).strip()
            labels.append(translate_category(category) if category else "")
    if len(labels) != len(paths) or any(not label for label in labels):
        return [dataset_label(path) for path in paths]
    return labels


def is_vehicle_dataset_dir(dataset_dir: Path) -> bool:
    name = dataset_dir.name.lower().replace(" ", "_")
    return "used_car" in name or "vehicle" in name


def vehicle_category_labels(paths: list[Path]) -> list[str]:
    labels: list[str] = []
    for index, path in enumerate(paths, start=1):
        match = re.search(r"(\d+)(?!.*\d)", path.stem)
        label_id = int(match.group(1)) if match else index
        labels.append(f"车辆数据集 {label_id:02d}")
    return labels


def parse_raw_float(value: object) -> float | None:
    try:
        result = float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def transformed_key(point: np.ndarray | tuple[float, ...]) -> tuple[float, ...]:
    return tuple(round(float(value), 10) for value in point)


def build_amazon_display_values(dataset_dir: Path, labels: list[str]) -> list[dict[tuple[float, ...], dict[str, Any]]]:
    summary_path = dataset_dir / "dataset_summary.csv"
    if not AMAZON_RAW_CSV.exists() or not summary_path.exists():
        return [{} for _ in labels]

    categories: list[str] = []
    with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            category = str(row.get("category", "")).strip()
            if category:
                categories.append(category)
    if not categories:
        return [{} for _ in labels]
    category_to_dataset = {category: idx for idx, category in enumerate(categories)}

    mins = [float("inf"), float("inf"), float("inf")]
    maxs = [float("-inf"), float("-inf"), float("-inf")]
    with AMAZON_RAW_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = valid_amazon_row(row)
            if parsed is None:
                continue
            _category, (price, _stars, reviews, bought) = parsed
            logged = [math.log1p(price), math.log1p(reviews), math.log1p(bought)]
            for i, value in enumerate(logged):
                mins[i] = min(mins[i], value)
                maxs[i] = max(maxs[i], value)

    def scale(value: float, index: int) -> float:
        logged = math.log1p(value)
        if maxs[index] <= mins[index]:
            return 1.0
        return (logged - mins[index]) / (maxs[index] - mins[index])

    display_maps: list[dict[tuple[float, ...], dict[str, Any]]] = [{} for _ in categories]
    with AMAZON_RAW_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = valid_amazon_row(row)
            if parsed is None:
                continue
            category, (price, stars, reviews, bought) = parsed
            dataset_id = category_to_dataset.get(category)
            if dataset_id is None:
                continue
            point = (
                1.0 - scale(price, 0),
                stars / 5.0,
                scale(reviews, 1),
                scale(bought, 2),
            )
            key = transformed_key(point)
            display_maps[dataset_id].setdefault(
                key,
                {
                    "title": str(row.get("title", "")).strip() or "未命名商品",
                    "title_zh": translate_title(row.get("title", "")),
                    "values": [
                        f"GBP {price:,.2f}",
                        f"{stars:.1f} / 5",
                        f"{int(round(reviews)):,}",
                        f"{int(round(bought)):,}",
                    ],
                },
            )
    return display_maps


def build_vehicle_display_values(datasets: list[np.ndarray]) -> list[dict[tuple[float, ...], dict[str, Any]]]:
    display_maps: list[dict[tuple[float, ...], dict[str, Any]]] = []
    for dataset_id, data in enumerate(datasets, start=1):
        mapping: dict[tuple[float, ...], dict[str, Any]] = {}
        for point_index, row in enumerate(data):
            values = [float(value) for value in row]
            title = f"二手车候选 {dataset_id:02d}-{point_index}"
            mapping.setdefault(
                transformed_key(tuple(values)),
                {
                    "title": title,
                    "title_zh": title,
                    "values": [f"{value:.3f}" for value in values],
                },
            )
        display_maps.append(mapping)
    return display_maps


@dataclass
class ItemView:
    dataset_id: int
    dataset_name: str
    point_index: int
    values: list[float]
    title: str
    title_zh: str
    display_values: list[str]


@dataclass
class QuestionView:
    method: str
    task_index: int
    task_total: int
    question_index: int
    item_a: ItemView
    item_b: ItemView
    active_utility_count: int
    candidate_count: int


@dataclass
class MethodSnapshot:
    active_mask: np.ndarray
    constraints: list[list[float]]
    dataset_index: int
    finished: bool
    questions: int
    final_indices: dict[int, int]
    current_question: QuestionView | None
    last_question_at: float


@dataclass
class MethodState:
    method: str
    utilities: np.ndarray
    active_mask: np.ndarray
    constraints: list[list[float]] = field(default_factory=list)
    dataset_index: int = 0
    finished: bool = False
    questions: int = 0
    started_at: float = field(default_factory=time.time)
    last_question_at: float = field(default_factory=time.time)
    final_indices: dict[int, int] = field(default_factory=dict)
    current_question: QuestionView | None = None
    history: list[MethodSnapshot] = field(default_factory=list)


@dataclass
class StudySession:
    session_id: str
    participant_id: str
    methods: list[str]
    method_index: int = 0
    method_state: MethodState | None = None
    created_at: float = field(default_factory=time.time)


class StudyEngine:
    def __init__(
        self,
        data: AmazonStudyData | None = None,
        result_dir: Path = DEFAULT_RESULT_DIR,
        utility_sample_count: int = 4096,
        seed: int = 20260601,
        max_questions_per_method: int = 60,
    ) -> None:
        self.data = data or AmazonStudyData.load()
        self.result_dir = result_dir
        self.utility_sample_count = int(utility_sample_count)
        self.seed = int(seed)
        self.max_questions_per_method = int(max_questions_per_method)
        self.sessions: dict[str, StudySession] = {}

    def create_session(self, participant_id: str, methods: list[str], shuffle_methods: bool) -> StudySession:
        methods = [method for method in methods if method in METHODS]
        if not methods:
            methods = list(METHODS)
        if shuffle_methods:
            digest = hashlib.sha256(participant_id.encode("utf-8")).digest()
            stable_offset = int.from_bytes(digest[:8], "little") % 100_000
            rng = np.random.default_rng(self.seed + stable_offset)
            methods = [methods[i] for i in rng.permutation(len(methods))]
        session = StudySession(
            session_id=uuid.uuid4().hex,
            participant_id=participant_id.strip() or "anonymous",
            methods=methods,
        )
        self.sessions[session.session_id] = session
        self._start_current_method(session)
        return session

    def get_session(self, session_id: str) -> StudySession | None:
        return self.sessions.get(session_id)

    def _sample_utilities(self, method: str, participant_id: str) -> np.ndarray:
        salt = sum(ord(ch) for ch in method + participant_id)
        rng = np.random.default_rng(self.seed + salt)
        return rng.dirichlet(np.ones(self.data.dim), size=self.utility_sample_count)

    def _start_current_method(self, session: StudySession) -> None:
        if session.method_index >= len(session.methods):
            session.method_state = None
            return
        method = session.methods[session.method_index]
        utilities = self._sample_utilities(method, session.participant_id)
        session.method_state = MethodState(
            method=method,
            utilities=utilities,
            active_mask=np.ones(len(utilities), dtype=bool),
        )

    def current_question(self, session: StudySession) -> QuestionView | None:
        state = session.method_state
        if state is None or state.finished:
            return None
        if state.current_question is None:
            state.current_question = self._next_question(session, state)
        return state.current_question

    def answer(self, session: StudySession, choice: str) -> None:
        state = session.method_state
        if state is None or state.current_question is None:
            return
        question = state.current_question
        state.history.append(
            MethodSnapshot(
                active_mask=state.active_mask.copy(),
                constraints=[list(item) for item in state.constraints],
                dataset_index=int(state.dataset_index),
                finished=bool(state.finished),
                questions=int(state.questions),
                final_indices=dict(state.final_indices),
                current_question=state.current_question,
                last_question_at=float(state.last_question_at),
            )
        )
        choice = "A" if choice.upper() == "A" else "B"
        preferred = question.item_a if choice == "A" else question.item_b
        rejected = question.item_b if choice == "A" else question.item_a

        preferred_point = self.data.datasets[preferred.dataset_id][preferred.point_index]
        rejected_point = self.data.datasets[rejected.dataset_id][rejected.point_index]
        normal = preferred_point - rejected_point
        scores = state.utilities @ normal
        next_mask = state.active_mask & (scores >= -1e-12)
        if not np.any(next_mask):
            # Keep the previous range if sampling resolution cannot represent the answer.
            next_mask = state.active_mask
        state.active_mask = next_mask
        state.constraints.append(normal.tolist())
        state.questions += 1

        now = time.time()
        append_csv(
            self.result_dir / "interactions.csv",
            {
                "timestamp": now,
                "session_id": session.session_id,
                "participant_id": session.participant_id,
                "method": state.method,
                "method_index": session.method_index,
                "question_index": state.questions,
                "dataset_id": question.item_a.dataset_id,
                "dataset_name": question.item_a.dataset_name,
                "item_a_index": question.item_a.point_index,
                "item_b_index": question.item_b.point_index,
                "choice": choice,
                "answer_time": now - state.last_question_at,
                "active_utility_count": int(np.sum(state.active_mask)),
                "candidate_count": question.candidate_count,
            },
        )
        state.current_question = None
        state.last_question_at = now

    def can_undo(self, session: StudySession | None) -> bool:
        state = session.method_state if session is not None else None
        return bool(state is not None and state.history)

    def undo(self, session: StudySession) -> bool:
        state = session.method_state
        if state is None or not state.history:
            return False
        snapshot = state.history.pop()
        state.active_mask = snapshot.active_mask.copy()
        state.constraints = [list(item) for item in snapshot.constraints]
        state.dataset_index = int(snapshot.dataset_index)
        state.finished = bool(snapshot.finished)
        state.questions = int(snapshot.questions)
        state.final_indices = dict(snapshot.final_indices)
        state.current_question = snapshot.current_question
        state.last_question_at = time.time()
        return True

    def finish_survey(self, session: StudySession, survey: dict[str, int]) -> None:
        state = session.method_state
        if state is None:
            return
        append_csv(
            self.result_dir / "surveys.csv",
            {
                "participant_id": session.participant_id,
                "questions": state.questions,
                "annoyance": survey.get("annoyance", ""),
                "satisfaction": survey.get("satisfaction", ""),
            },
        )
        session.method_index += 1
        self._start_current_method(session)

    def is_complete(self, session: StudySession) -> bool:
        return session.method_index >= len(session.methods)

    def final_recommendations(self, state: MethodState) -> list[ItemView]:
        self._finish_remaining_final_indices(state)
        result = []
        for dataset_id, point_index in sorted(state.final_indices.items()):
            result.append(self._item_view(dataset_id, point_index))
        return result

    def _finish_remaining_final_indices(self, state: MethodState) -> None:
        for dataset_id, data in enumerate(self.data.datasets):
            if dataset_id in state.final_indices:
                continue
            candidates = self._candidate_indices(data, state.active_mask, state.utilities)
            state.final_indices[dataset_id] = int(candidates[0]) if len(candidates) else int(np.argmax(data @ np.mean(state.utilities[state.active_mask], axis=0)))

    def _next_question(self, session: StudySession, state: MethodState) -> QuestionView | None:
        if state.questions >= self.max_questions_per_method:
            state.finished = True
            return None

        if state.method.startswith("SUR-"):
            pair = self._next_sur_pair(state)
        else:
            pair = self._next_sharedq_pair(state)

        if pair is None:
            state.finished = True
            return None
        dataset_id, left, right, candidate_count = pair
        return QuestionView(
            method=state.method,
            task_index=session.method_index + 1,
            task_total=len(session.methods),
            question_index=state.questions + 1,
            item_a=self._item_view(dataset_id, left),
            item_b=self._item_view(dataset_id, right),
            active_utility_count=int(np.sum(state.active_mask)),
            candidate_count=int(candidate_count),
        )

    def _next_sur_pair(self, state: MethodState) -> tuple[int, int, int, int] | None:
        while state.dataset_index < len(self.data.datasets):
            dataset_id = state.dataset_index
            data = self.data.datasets[dataset_id]
            candidates = self._candidate_indices(data, state.active_mask, state.utilities)
            if len(candidates) <= 1:
                if len(candidates) == 1:
                    state.final_indices[dataset_id] = int(candidates[0])
                state.dataset_index += 1
                continue
            pair = self._select_pair_for_method(
                state.method,
                data,
                candidates,
                state.active_mask,
                state.utilities,
                dataset_id,
            )
            if pair is None:
                state.dataset_index += 1
                continue
            return dataset_id, pair[0], pair[1], len(candidates)
        return None

    def _next_sharedq_pair(self, state: MethodState) -> tuple[int, int, int, int] | None:
        best: tuple[float, int, int, int, int] | None = None
        total_candidates = 0
        for dataset_id, data in enumerate(self.data.datasets):
            candidates = self._candidate_indices(data, state.active_mask, state.utilities)
            total_candidates += len(candidates)
            if len(candidates) <= 1:
                if len(candidates) == 1:
                    state.final_indices[dataset_id] = int(candidates[0])
                continue
            pair = self._best_balanced_pair(data, candidates, state.active_mask, state.utilities)
            if pair is None:
                continue
            left, right, score = pair
            if best is None or score > best[0]:
                best = (score, dataset_id, left, right, len(candidates))
        if best is None:
            return None
        _score, dataset_id, left, right, candidate_count = best
        return dataset_id, left, right, max(candidate_count, total_candidates)

    def _candidate_indices(self, data: np.ndarray, active_mask: np.ndarray, utilities: np.ndarray) -> list[int]:
        active_utilities = utilities[active_mask]
        if len(active_utilities) == 0:
            active_utilities = utilities
        winners = np.argmax(data @ active_utilities.T, axis=0)
        return sorted(int(item) for item in np.unique(winners))

    def _select_pair_for_method(
        self,
        method: str,
        data: np.ndarray,
        candidates: list[int],
        active_mask: np.ndarray,
        utilities: np.ndarray,
        dataset_id: int,
    ) -> tuple[int, int] | None:
        if len(candidates) < 2:
            return None
        rng = np.random.default_rng(self.seed + dataset_id * 1009 + len(candidates) * 31)
        if method == "SUR-UH-Random":
            left, right = rng.choice(candidates, size=2, replace=False)
            return int(left), int(right)
        if method == "SUR-RH":
            return self._random_balanced_subset(data, candidates, active_mask, utilities, rng)
        if method == "SUR-UH-Simplex":
            return self._farthest_pair(data, candidates)
        return_pair = self._best_balanced_pair(data, candidates, active_mask, utilities)
        if return_pair is None:
            return None
        return return_pair[0], return_pair[1]

    def _pair_preference_counts(
        self,
        data: np.ndarray,
        left: int,
        right: int,
        active_mask: np.ndarray,
        utilities: np.ndarray,
    ) -> tuple[int, int]:
        active = utilities[active_mask]
        if len(active) == 0:
            active = utilities
        scores = active @ (data[left] - data[right])
        positive = int(np.sum(scores >= 0))
        negative = int(len(active) - positive)
        return positive, negative

    def _best_balanced_pair(
        self,
        data: np.ndarray,
        candidates: list[int],
        active_mask: np.ndarray,
        utilities: np.ndarray,
    ) -> tuple[int, int, float] | None:
        best: tuple[int, int, float] | None = None
        candidate_pairs = self._candidate_pairs(candidates, max_pairs=400)
        for left, right in candidate_pairs:
            positive, negative = self._pair_preference_counts(data, left, right, active_mask, utilities)
            score = min(positive, negative)
            if best is None or score > best[2]:
                best = (left, right, float(score))
        return best

    def _random_balanced_subset(
        self,
        data: np.ndarray,
        candidates: list[int],
        active_mask: np.ndarray,
        utilities: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[int, int] | None:
        pairs = self._candidate_pairs(candidates, max_pairs=80, rng=rng)
        if not pairs:
            return None
        scored = []
        for left, right in pairs:
            positive, negative = self._pair_preference_counts(data, left, right, active_mask, utilities)
            scored.append((min(positive, negative), left, right))
        scored.sort(reverse=True)
        _score, left, right = scored[0]
        return left, right

    def _farthest_pair(self, data: np.ndarray, candidates: list[int]) -> tuple[int, int] | None:
        best: tuple[float, int, int] | None = None
        for left, right in self._candidate_pairs(candidates, max_pairs=500):
            dist = float(np.linalg.norm(data[left] - data[right]))
            if best is None or dist > best[0]:
                best = (dist, left, right)
        if best is None:
            return None
        return best[1], best[2]

    def _candidate_pairs(
        self,
        candidates: list[int],
        max_pairs: int,
        rng: np.random.Generator | None = None,
    ) -> list[tuple[int, int]]:
        if len(candidates) < 2:
            return []
        pairs: list[tuple[int, int]] = []
        total = len(candidates) * (len(candidates) - 1) // 2
        if total <= max_pairs:
            for i, left in enumerate(candidates):
                for right in candidates[i + 1:]:
                    pairs.append((int(left), int(right)))
            return pairs
        rng = rng or np.random.default_rng(self.seed + len(candidates))
        seen: set[tuple[int, int]] = set()
        while len(pairs) < max_pairs and len(seen) < total:
            left, right = rng.choice(candidates, size=2, replace=False)
            pair = tuple(sorted((int(left), int(right))))
            if pair in seen:
                continue
            seen.add(pair)
            pairs.append(pair)
        return pairs

    def _item_view(self, dataset_id: int, point_index: int) -> ItemView:
        values = [float(x) for x in self.data.datasets[dataset_id][point_index]]
        display = self.data.display_values[dataset_id].get(
            transformed_key(tuple(values)),
            {
                "title": f"商品 #{point_index}",
                "values": [f"{value:.3f}" for value in values],
            },
        )
        return ItemView(
            dataset_id=dataset_id,
            dataset_name=self.data.labels[dataset_id],
            point_index=int(point_index),
            values=values,
            title=str(display.get("title", f"商品 #{point_index}")),
            title_zh=str(display.get("title_zh", display.get("title", f"商品 #{point_index}"))),
            display_values=list(display.get("values", [f"{value:.3f}" for value in values])),
        )
