"""Math utils"""


def is_kronecker_delta(vector):
    """
    Check if vector is a kronecker delta.

    Args:
        vector: iterable of numbers
    Returns:
        bool, True if vector is a kronecker delta vector, False otherwise.
        In belief propagation, specific evidence (variable is directly observed)
        is a kronecker delta vector, but virtual evidence is not.
    """
    count = 0
    for x in vector:
        if x == 1:
            count += 1
        elif x != 0:
            return False

    if count == 1:
        return True
    else:
        return False
