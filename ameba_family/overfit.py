import matplotlib.pyplot as plt

def show_losses_paths():


    # plt.xkcd()
    infile = open('series.pkl', 'rb')
    series_of_losses = pkl.load(infile)
    infile.close()
    x = [i for i in range(len(series_of_losses[0]))]

    for i in range(len(series_of_losses)):
        plt.plot(x, series_of_losses[i], label=str(i))

    leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=False)

    leg.get_frame().set_alpha(0.6)
    plt.show()