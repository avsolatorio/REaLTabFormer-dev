import uuid
from dataclasses import dataclass, fields

TEACHER_FORCING_PRE = "_TEACHERFORCING"
SPECIAL_COL_SEP = "___"
NUMERIC_NA_TOKEN = "@"
INVALID_NUMS_RE = r"[^\-.0-9]"


@dataclass(frozen=True)
class TabularArtefact:
    best_disc_model: str = "best-disc-model"
    mean_best_disc_model: str = "mean-best-disc-model"
    not_best_disc_model: str = "not-best-disc-model"
    last_epoch_model: str = "last-epoch-model"

    @staticmethod
    def artefacts():
        return [field.default for field in fields(TabularArtefact)]


@dataclass(frozen=True)
class ModelFileName:
    rtf_config_json: str = "rtf_config.json"
    rtf_model_pt: str = "rtf_model.pt"

    @staticmethod
    def names():
        return [field.default for field in fields(ModelFileName)]


@dataclass(frozen=True)
class ModelType:
    tabular: str = "tabular"
    relational: str = "relational"

    @staticmethod
    def types():
        return [field.default for field in fields(ModelType)]


@dataclass(frozen=True)
class ColDataType:
    NUMERIC: str = "NUMERIC"
    DATETIME: str = "DATETIME"
    CATEGORICAL: str = "CATEGORICAL"

    @staticmethod
    def types():
        return [field.default for field in fields(ColDataType)]


@dataclass(frozen=True)
class SpecialTokens:
    UNK: str = "[UNK]"
    SEP: str = "[SEP]"
    PAD: str = "[PAD]"
    CLS: str = "[CLS]"
    MASK: str = "[MASK]"
    BOS: str = "[BOS]"
    EOS: str = "[EOS]"
    BMEM: str = "[BMEM]"
    EMEM: str = "[EMEM]"
    RMASK: str = "[RMASK]"
    SPTYPE: str = "[SPTYPE]"

    @staticmethod
    def tokens():
        return [field.default for field in fields(SpecialTokens)]


def get_uuid():
    return uuid.uuid4().hex
