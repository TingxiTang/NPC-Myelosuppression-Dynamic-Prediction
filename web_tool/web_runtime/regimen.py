"""Frozen treatment-regimen catalog and one-cycle scenario adapter."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


EXPECTED_CATALOG_SHA256 = "357fd0b081d974b4fc76a8c5ce577d2ea2d6b281bda0d4754ae85ac26e1052b5"
DEFAULT_CATALOG_PATH = Path(__file__).with_name("assets") / "drug_category_index_clean.csv"
REQUIRED_REGIMEN_COLUMNS = (
    "drug_id",
    "category_id",
    "is_chemo",
    "is_target",
    "is_immuno",
    "is_rt",
)


class RegimenValidationError(ValueError):
    """Raised when a treatment scenario cannot satisfy the frozen mapping."""


@dataclass(frozen=True)
class RegimenEntry:
    drug_id: int
    drug_name: str
    category_id: int
    category_name: str

    @property
    def display_name(self) -> str:
        if self.drug_id == 2:
            return "放射治疗（RT）"
        return self.drug_name


@dataclass(frozen=True)
class RegimenSelection:
    drug_ids: tuple[int, ...]
    category_ids: tuple[int, ...]
    display_names: tuple[str, ...]
    is_chemo: int
    is_target: int
    is_immuno: int
    is_rt: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_regimen_catalog(
    drug_vocab: Sequence[int],
    category_vocab: Sequence[int],
    *,
    path: str | Path = DEFAULT_CATALOG_PATH,
) -> tuple[RegimenEntry, ...]:
    """Load the audited legacy mapping and restrict it to the frozen encoder vocabularies."""

    catalog_path = Path(path)
    if not catalog_path.is_file() or catalog_path.is_symlink():
        raise RegimenValidationError("治疗方案字典缺失")
    if _sha256(catalog_path) != EXPECTED_CATALOG_SHA256:
        raise RegimenValidationError("治疗方案字典校验失败")

    expected_drugs = {int(value) for value in drug_vocab}
    expected_categories = {int(value) for value in category_vocab}
    if not expected_drugs or not expected_categories:
        raise RegimenValidationError("冻结治疗编码词表为空")

    entries: list[RegimenEntry] = []
    seen_ids: set[int] = set()
    seen_names: set[str] = set()
    with catalog_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        expected_columns = {
            "Drug_ID",
            "Drug_Name(Cn)",
            "Drug_Name(En)",
            "Category_ID",
            "Category_Name",
        }
        if set(reader.fieldnames or ()) != expected_columns:
            raise RegimenValidationError("治疗方案字典列结构异常")
        for row in reader:
            try:
                drug_id = int(row["Drug_ID"])
                category_id = int(row["Category_ID"])
            except (TypeError, ValueError) as error:
                raise RegimenValidationError("治疗方案字典含无效编码") from error
            if drug_id not in expected_drugs:
                continue
            drug_name = str(row["Drug_Name(Cn)"]).strip()
            category_name = str(row["Category_Name"]).strip()
            if (
                not drug_name
                or not category_name
                or drug_id in seen_ids
                or drug_name in seen_names
                or category_id not in expected_categories
            ):
                raise RegimenValidationError("治疗方案字典与冻结编码不一致")
            seen_ids.add(drug_id)
            seen_names.add(drug_name)
            entries.append(
                RegimenEntry(
                    drug_id=drug_id,
                    drug_name=drug_name,
                    category_id=category_id,
                    category_name=category_name,
                )
            )

    if seen_ids != expected_drugs:
        raise RegimenValidationError("治疗方案字典未覆盖全部冻结药物编码")
    return tuple(sorted(entries, key=lambda entry: entry.drug_id))


def selected_drug_ids(
    frame: pd.DataFrame,
    catalog: Sequence[RegimenEntry],
) -> tuple[int, ...]:
    """Recover the currently encoded regimen without silently dropping unknown IDs."""

    if not isinstance(frame, pd.DataFrame) or len(frame) != 1 or "drug_id" not in frame:
        raise RegimenValidationError("治疗周期输入缺少用药方案")
    raw = frame.iloc[0]["drug_id"]
    if not isinstance(raw, (list, tuple, set)):
        raise RegimenValidationError("用药方案必须是编码列表")
    known = {entry.drug_id for entry in catalog}
    selected: list[int] = []
    for value in raw:
        if isinstance(value, bool):
            raise RegimenValidationError("用药方案含无效编码")
        try:
            drug_id = int(value)
        except (TypeError, ValueError) as error:
            raise RegimenValidationError("用药方案含无效编码") from error
        if drug_id not in known:
            raise RegimenValidationError(f"用药编码 {drug_id} 不在冻结字典中")
        if drug_id not in selected:
            selected.append(drug_id)
    if not selected:
        raise RegimenValidationError("请至少选择一个本次治疗项目")
    return tuple(selected)


def apply_regimen(
    frame: pd.DataFrame,
    selected_ids: Iterable[int],
    catalog: Sequence[RegimenEntry],
) -> tuple[pd.DataFrame, RegimenSelection]:
    """Replace only current-regimen fields while preserving baseline and cumulative exposure."""

    if not isinstance(frame, pd.DataFrame) or len(frame) != 1:
        raise RegimenValidationError("治疗周期输入必须仅含一行")
    missing = [column for column in REQUIRED_REGIMEN_COLUMNS if column not in frame]
    if missing:
        raise RegimenValidationError("治疗周期输入缺少方案字段：" + ", ".join(missing))

    by_id = {entry.drug_id: entry for entry in catalog}
    normalized: list[int] = []
    for value in selected_ids:
        if isinstance(value, bool):
            raise RegimenValidationError("用药方案含无效编码")
        try:
            drug_id = int(value)
        except (TypeError, ValueError) as error:
            raise RegimenValidationError("用药方案含无效编码") from error
        if drug_id not in by_id:
            raise RegimenValidationError(f"用药编码 {drug_id} 不在冻结字典中")
        if drug_id not in normalized:
            normalized.append(drug_id)
    if not normalized:
        raise RegimenValidationError("请至少选择一个本次治疗项目")

    entries = [by_id[drug_id] for drug_id in normalized]
    category_ids = tuple(sorted({entry.category_id for entry in entries}))
    is_chemo = int(any(entry.category_name.startswith("Chemo_") for entry in entries))
    is_target = int(any(entry.category_name.startswith("Target_") for entry in entries))
    is_immuno = int(any(entry.category_name.startswith("Immuno_") for entry in entries))
    is_rt = int(any(entry.category_name == "RT" for entry in entries))
    selection = RegimenSelection(
        drug_ids=tuple(normalized),
        category_ids=category_ids,
        display_names=tuple(entry.display_name for entry in entries),
        is_chemo=is_chemo,
        is_target=is_target,
        is_immuno=is_immuno,
        is_rt=is_rt,
    )

    updated = frame.copy()
    for column in ("drug_id", "category_id"):
        updated[column] = updated[column].astype(object)
    updated.at[updated.index[0], "drug_id"] = list(selection.drug_ids)
    updated.at[updated.index[0], "category_id"] = list(selection.category_ids)
    updated.at[updated.index[0], "is_chemo"] = selection.is_chemo
    updated.at[updated.index[0], "is_target"] = selection.is_target
    updated.at[updated.index[0], "is_immuno"] = selection.is_immuno
    updated.at[updated.index[0], "is_rt"] = selection.is_rt
    return updated, selection
