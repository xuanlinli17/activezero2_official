from active_zero2.models.cgi_stereo.cgi_stereo import CGI_Stereo


def build_model(cfg):
    model = CGI_Stereo(
        maxdisp=cfg.CGIStereo.MAX_DISP,
        disparity_mode=cfg.CGIStereo.DISPARITY_MODE,
        loglinear_disp_min_depth=cfg.CGIStereo.LOGLINEAR_DISP_MIN_DEPTH,
        loglinear_disp_max_depth=cfg.CGIStereo.LOGLINEAR_DISP_MAX_DEPTH,
        loglinear_disp_c=cfg.CGIStereo.LOGLINEAR_DISP_C,
        predict_normal=cfg.CGIStereo.PREDICT_NORMAL,
        predict_normal_v2=cfg.CGIStereo.PREDICT_NORMAL_V2,
    )
    return model
