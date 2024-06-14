def gen_countinglist(proof_element, xy, x1x, x):
    not_in = 0
    t = 1
    while (t<=len())

def gen_transfer_entropy(binned_spk_trains):
    num_cols = binned_spk_trains.shape[1]
    num_rows = binned_spk_trains.shape[0]
    results = np.zeros((num_rows, num_rows))
    m = 0
    k = 2
    steps = 1000
    for z in range(num_rows):
        for p in range(num_rows - z):
            for s in range(num_cols - k):
                words_k_x = 0
                words_k_y1 = 0
                words_k_x1 = binned_spk_trains(z, k + s)


                # reverse
                wordsr_k_x = 0
                wordsr_k_y1 = 0
                wordsr_k_x1 = binned_spk_trains(z + p, k + s)

                L = 1
                for o in range(k - 1, 0):
                    words_k_x = words_k_x + binned_spk_trains(z, s + o) * L
                    words_k_y1 = words_k_y1 + binned_spk_trains(z + p, s + o + 1) * L
                    L *= 10

                xy1x1 = words_k_x * 10 ** (k + 1) + words_k_y1 * 10 ** (1) + words_k_x1
                xy1 = words_k_x * 10**k + words_k_y1
                x1x = words_k_x1 * 10**k + words_k_x

                L = 1
                for o in range(k - 1, 0):
                    wordsr_k_x = wordsr_k_x + binned_spk_trains(z, s + o) * L
                    wordsr_k_y1 = wordsr_k_y1 + binned_spk_trains(z + p, s + o + 1) * L
                    L *= 10

                rxy1x1 = (
                    wordsr_k_x * 10 ** (k + 1) + wordsr_k_y1 * 10 ** (1) + wordsr_k_x1
                )
                rxy1 = wordsr_k_x * 10**k + wordsr_k_y1
                roundx1x = wordsr_k_x1 * 10**k + wordsr_k_x
