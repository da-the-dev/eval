import re


def preprocess(text: str) -> list[str]:
    return re.sub(r'[^A-z\s\d][\\\^]?', '', text).lower().split()


def _f1_score(precision: float, recall: float) -> float:

    if precision + recall == 0:
        raise ValueError(f'Precision + recall cannot be 0, division by zero. Precision: {precision}, recall: {recall}')

    return 2 * precision * recall / (precision + recall)


def _create_ngrams(tokens: list[str], n: int) -> set[tuple]:
    ngrams = set()
    for i in range(0, len(tokens) + 1 - n):
        start_ngram = i
        end_ngram = i + n

        ngrams.add(tuple(tokens[start_ngram:end_ngram]))

    return ngrams


def rouge_n(candidate_tokens: list[str], reference_tokens: list[str], n: int) -> dict[str, float]:
    """
    Compute ROUGE-N

    Returns:
        dict[str, float]: precision, recall, and f1 as a dict. F1 is the ROUGE-N score
    """
    candidate_ngram = _create_ngrams(candidate_tokens, n)
    reference_ngram = _create_ngrams(reference_tokens, n)

    overlap = candidate_ngram.intersection(reference_ngram)

    p = len(overlap) / len(candidate_ngram) if len(candidate_ngram) > 0 else 0.0
    r = len(overlap) / len(reference_ngram) if len(reference_ngram) > 0 else 0.0

    f1 = _f1_score(p, r)

    return {'precision': p, 'recall': r, 'f1': f1}


def _lcs_length(x: list[str], y: list[str]) -> int:
    m, n = len(x), len(y)

    # Initialize the DP grid with zeros.
    # grid[i][j] will store the length of LCS of X[:i] and Y[:j]
    grid = [[0] * (n + 1) for _ in range(m + 1)]

    # Build the grid in a bottom-up fashion
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                # If the current tokens match, extend the LCS from the previous diagonal cell.
                grid[i][j] = grid[i - 1][j - 1] + 1
            else:
                # If they don't match, take the maximum LCS length from the top or left cell.
                grid[i][j] = max(grid[i - 1][j], grid[i][j - 1])

    # The value in the bottom-right corner is the length of the LCS for the entire sequences.
    return grid[m][n]


def rouge_l(candidate_tokens: list[str], reference_tokens: list[str]) -> dict[str, float]:
    """
    Compute ROUGE-L

    Returns:
        dict[str, float]: precision, recall, and f1 as a dict. F1 is the ROUGE-L score
    """
    lcs = _lcs_length(candidate_tokens, reference_tokens)

    p = lcs / len(candidate_tokens)
    r = lcs / len(reference_tokens)

    f1 = _f1_score(p, r)

    return {'precision': p, 'recall': r, 'f1': f1}
