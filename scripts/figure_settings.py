import pylab
palette = ["#2176AE",  "#B98E00", "#FE6847", "#0CCA4A"]
linestyles = ["-", "--", "-.", ":"]


def lighten(color, amount=0.25):
    import matplotlib.colors as mcolors
    c = mcolors.to_rgb(color)
    lightened = tuple([max(0, min(1, c[i] + amount)) for i in range(3)])
    hexd = '#%02x%02x%02x' % tuple(int(255 * x) for x in lightened)
    
    return hexd

def set_fonts(extra_params={}):
    params = {
        # "font.family": "Serif",
        "font.sans-serif": ["Cantarell", "DejaVu Sans", "Lucida Grande", "Verdana"],
        # "mathtext.fontset": "cm",
        "legend.fontsize": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.titlesize": 12,
        "font.size": 14,
    }
    for key, value in extra_params.items():
        params[key] = value
    pylab.rcParams.update(params)
    

theta = {
    "copy" : "rho", 
    "extant" : "gamma", 
    "novel" : "eta"
}
    

# import matplotlib.font_manager
# fpaths = matplotlib.font_manager.findSystemFonts()

# for i in fpaths:
#     try: 
#         f = matplotlib.font_manager.get_font(i)
#         print(f.family_name)
#     except:
#         pass