from active_zero2.models.cgi_stereo.cgi_stereo import CGI_Stereo


def build_model(cfg):
    model = CGI_Stereo(
        maxdisp=cfg.CGIStereo.MAX_DISP,
        disparity_mode=cfg.CGIStereo.DISPARITY_MODE,
    )
    return model
