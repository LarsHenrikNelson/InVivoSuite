import numpy as np


class WordAnalyzer:
    """
    Efficient word analysis for large neural populations using byte encoding.
    """

    def __init__(self, raster):
        """
        Parameters
        ----------
        raster : ndarray
            Binary array (neurons × timepoints), dtype should be uint8 or bool
        """
        self.raster = np.asarray(raster, dtype=np.uint8)
        self.n_neurons, self.n_timepoints = raster.shape

        # Precompute firing rates and log probabilities
        self.firing_rates = np.clip(raster.mean(axis=1), 1e-10, 1 - 1e-10)
        self.log_p = np.log(self.firing_rates)
        self.log_1_minus_p = np.log(1 - self.firing_rates)

        # Cache
        self._packed = None
        self._analysis = None

    @property
    def packed(self):
        """Packed bit representation (timepoints × n_bytes)."""
        if self._packed is None:
            self._packed = np.packbits(self.raster.T, axis=1)
        return self._packed

    def _pack_raster(self, raster):
        """Pack any raster to bytes."""
        return np.packbits(raster.T, axis=1)

    def _count_packed(self, packed):
        """Count unique rows in packed array."""
        row_size = packed.shape[1]
        packed_view = (
            np.ascontiguousarray(packed)
            .view(dtype=np.dtype((np.void, row_size)))
            .ravel()
        )

        unique_views, inverse, counts = np.unique(
            packed_view, return_inverse=True, return_counts=True
        )

        # Convert to bytes for dictionary keys
        unique_bytes = [v.tobytes() for v in unique_views]

        return unique_bytes, inverse, counts

    def bytes_to_bits(self, word_bytes):
        """Convert bytes back to bit pattern."""
        packed = np.frombuffer(word_bytes, dtype=np.uint8)
        bits = np.unpackbits(packed)[: self.n_neurons]
        return bits

    def analyze(self, force=False):
        """
        Full analysis of word frequencies.

        Returns
        -------
        dict with analysis results
        """
        if self._analysis is not None and not force:
            return self._analysis

        unique_words, inverse, counts = self._count_packed(self.packed)
        n_unique = len(unique_words)

        # Convert all unique words to bit matrix
        bit_matrix = np.array([self.bytes_to_bits(w) for w in unique_words])

        # Vectorized log expected probability
        log_p_expected = bit_matrix @ self.log_p + (1 - bit_matrix) @ self.log_1_minus_p

        # Observed
        log_p_observed = np.log(counts / self.n_timepoints)

        # Statistics
        log_ratios = log_p_observed - log_p_expected
        expected_counts = self.n_timepoints * np.exp(log_p_expected)

        variance = expected_counts * (1 - np.exp(log_p_expected))
        std = np.sqrt(np.maximum(variance, 1e-10))
        z_scores = (counts - expected_counts) / std

        self._analysis = {
            "unique_words": unique_words,
            "bit_matrix": bit_matrix,
            "inverse": inverse,
            "counts": counts,
            "expected_counts": expected_counts,
            "log_p_observed": log_p_observed,
            "log_p_expected": log_p_expected,
            "log_ratios": log_ratios,
            "z_scores": z_scores,
            "n_active": bit_matrix.sum(axis=1),
            "n_unique": n_unique,
        }

        return self._analysis

    def _shuffle_raster(self, rng):
        """Shuffle each neuron's spike train independently."""
        shuffled = np.empty_like(self.raster)
        for i in range(self.n_neurons):
            shuffled[i] = rng.permutation(self.raster[i])
        return shuffled

    def _shuffle_raster_fast(self, rng):
        """
        Faster shuffling using argsort trick.
        Shuffles all neurons at once.
        """
        # Generate random values and argsort to get permutation indices
        random_vals = rng.random(self.raster.shape)
        sort_indices = np.argsort(random_vals, axis=1)

        # Apply permutation to each row
        row_indices = np.arange(self.n_neurons)[:, None]
        shuffled = self.raster[row_indices, sort_indices]

        return shuffled

    def permutation_test(
        self, n_permutations=1000, seed=None, top_n=None, z_threshold=None
    ):
        """
        Faster permutation test that only tracks summary statistics,
        not full null distributions.

        Parameters
        ----------
        n_permutations : int
            Number of shuffle iterations
        seed : int, optional
            Random seed
        top_n : int, optional
            Only test the top N most frequent observed words
        z_threshold : float, optional
            Only test words with |z_score| > threshold from independence model
        verbose : bool
            Print progress

        Returns
        -------
        dict with permutation test results
        """
        rng = np.random.default_rng(seed)

        # Get observed analysis
        analysis = self.analyze()
        observed_words = analysis["unique_words"]
        observed_counts = analysis["counts"]

        # Filter words to test
        test_mask = np.ones(len(observed_words), dtype=bool)

        if z_threshold is not None:
            test_mask &= np.abs(analysis["z_scores"]) > z_threshold

        if top_n is not None:
            # Get indices of top_n most frequent
            top_indices = np.argsort(observed_counts)[-top_n:]
            top_mask = np.zeros(len(observed_words), dtype=bool)
            top_mask[top_indices] = True
            test_mask &= top_mask

        test_indices = np.where(test_mask)[0]
        words_to_test = [observed_words[i] for i in test_indices]

        # Track running statistics (Welford's algorithm for numerical stability)
        n_words = len(words_to_test)
        word_to_idx = {w: i for i, w in enumerate(words_to_test)}

        null_sum = np.zeros(n_words)
        null_sum_sq = np.zeros(n_words)
        null_max = np.zeros(n_words, dtype=np.int32)
        null_min = np.full(n_words, self.n_timepoints, dtype=np.int32)
        count_exceeds = np.zeros(n_words, dtype=np.int32)  # For p_greater
        count_below = np.zeros(n_words, dtype=np.int32)  # For p_less

        obs_counts = np.array(
            [observed_counts[test_indices[i]] for i in range(n_words)]
        )

        # Run permutations
        for _ in range(n_permutations):
            # Shuffle and count
            shuffled = self._shuffle_raster_fast(rng)
            packed_shuffled = self._pack_raster(shuffled)
            null_words, _, null_word_counts = self._count_packed(packed_shuffled)

            # Build count lookup
            seen_this_perm = set()

            # Only iterate through null words that we're testing
            for null_word, count in zip(null_words, null_word_counts):
                if null_word in word_to_idx:
                    i = word_to_idx[null_word]
                    seen_this_perm.add(i)

                    null_sum[i] += count
                    null_sum_sq[i] += count * count
                    null_max[i] = max(null_max[i], count)
                    null_min[i] = min(null_min[i], count)

                    if count >= obs_counts[i]:
                        count_exceeds[i] += 1
                    if count <= obs_counts[i]:
                        count_below[i] += 1

            # Words not seen this permutation have count=0
            for i in range(n_words):
                if i not in seen_this_perm:
                    # count = 0, so null_sum and null_sum_sq unchanged
                    null_min[i] = 0
                    if 0 >= obs_counts[i]:
                        count_exceeds[i] += 1
                    if 0 <= obs_counts[i]:
                        count_below[i] += 1

        # Compute final statistics
        null_mean = null_sum / n_permutations
        null_var = (null_sum_sq / n_permutations) - (null_mean**2)
        null_std = np.sqrt(np.maximum(null_var, 0))

        z_scores = np.where(null_std > 0, (obs_counts - null_mean) / null_std, 0)

        p_greater = count_exceeds / n_permutations
        p_less = count_below / n_permutations
        p_two_sided = 2 * np.minimum(p_greater, p_less)
        p_two_sided = np.minimum(p_two_sided, 1.0)

        # Build results
        results = []
        for i, word in enumerate(words_to_test):
            results.append(
                {
                    "word": word,
                    "bits": self.bytes_to_bits(word),
                    "n_active": self.bytes_to_bits(word).sum(),
                    "observed": int(obs_counts[i]),
                    "null_mean": null_mean[i],
                    "null_std": null_std[i],
                    "null_min": int(null_min[i]),
                    "null_max": int(null_max[i]),
                    "z_score": z_scores[i],
                    "p_two_sided": p_two_sided[i],
                    "p_greater": p_greater[i],
                    "p_less": p_less[i],
                }
            )

        results.sort(key=lambda x: -abs(x["z_score"]))

        return {
            "results": results,
            "n_permutations": n_permutations,
            "n_tested": len(words_to_test),
        }

    def get_significant_words(self, method="independence", **kwargs):
        """
        Get significantly over/under-represented words.

        Parameters
        ----------
        method : str
            'independence' - compare to independence model (fast)
            'permutation' - use permutation test (slower, more robust)
        **kwargs :
            For 'independence': z_threshold (default 3.0)
            For 'permutation': n_permutations, p_threshold (default 0.05)

        Returns
        -------
        dict with 'synchrony' and 'anti_synchrony' word lists
        """
        if method == "independence":
            z_threshold = kwargs.get("z_threshold", 3.0)
            analysis = self.analyze()

            sync_mask = analysis["z_scores"] > z_threshold
            anti_mask = analysis["z_scores"] < -z_threshold

            def extract(mask):
                indices = np.where(mask)[0]
                return [
                    {
                        "word": analysis["unique_words"][i],
                        "bits": analysis["bit_matrix"][i],
                        "n_active": analysis["n_active"][i],
                        "count": analysis["counts"][i],
                        "expected": analysis["expected_counts"][i],
                        "z_score": analysis["z_scores"][i],
                    }
                    for i in indices
                ]

            return {
                "synchrony": sorted(extract(sync_mask), key=lambda x: -x["z_score"]),
                "anti_synchrony": sorted(
                    extract(anti_mask), key=lambda x: x["z_score"]
                ),
            }

        elif method == "permutation":
            n_perms = kwargs.get("n_permutations", 1000)
            p_threshold = kwargs.get("p_threshold", 0.05)

            perm_results = self.permutation_test(
                n_permutations=n_perms,
                z_threshold=kwargs.get("pre_filter_z", 2.0),
                verbose=kwargs.get("verbose", True),
            )

            synchrony = []
            anti_synchrony = []

            for r in perm_results["results"]:
                if r["p_greater"] < p_threshold:
                    anti_synchrony.append(r)  # Observed is rarely exceeded
                elif r["p_less"] < p_threshold:
                    synchrony.append(r)  # Observed rarely falls below null

            return {
                "synchrony": synchrony,
                "anti_synchrony": anti_synchrony,
                "permutation_details": perm_results,
            }
