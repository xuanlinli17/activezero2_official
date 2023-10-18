from active_zero2.models.psmnet.build_model import build_model as build_psmnet
from active_zero2.models.psmnet_edge_norm.build_model import build_model as build_psmnetedgenorm
from active_zero2.models.cgi_stereo.build_model import build_model as build_cgistereo

MODEL_LIST = (
    "PSMNet",
    "PSMNetEdgeNormal",
    "CGIStereo"
)


def build_model(cfg):
    if cfg.MODEL_TYPE == "PSMNet":
        model = build_psmnet(cfg)
    elif cfg.MODEL_TYPE == "PSMNetEdgeNormal":
        model = build_psmnetedgenorm(cfg)
    elif cfg.MODEL_TYPE == "CGIStereo":
        model = build_cgistereo(cfg)

    else:
        raise ValueError(f"Unexpected model type: {cfg.MODEL_TYPE}")

    return model
