"""Random math utils."""


def is_kronecker_delta(vector):
    """Returns True if vector is a kronecker delta vector, False otherwise.
    Specific evidence ('YES' or 'NO') is a kronecker delta vector, whereas
    virtual evidence ('MAYBE') is not.
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
