import matplotlib.pyplot as plt

def save_image(fname, image, dpi=300, pdf=True, png=False):
    # Normalize image between 0 and 1
    image = (image - image.min()) / (image.max() - image.min())
    
    for i in range(3):
        if pdf:
            plt.imsave(
                f"{fname}_{i}.pdf",
                image[i],
                vmin=0,
                vmax=1,
                dpi=dpi
            )

        if png:
            plt.imsave(
                f"{fname}_{i}.png",
                image[i],
                vmin=0,
                vmax=1,
                dpi=dpi
            )

def save_fig(fname, fig, dpi=300, pdf=True, png=False):    
    if pdf:
        fig.savefig(
            f"{fname}.pdf",
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.1
        )

    if png:
        fig.savefig(
            f"{fname}.png",
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.1
        )