"""
Prime number utilities and factorization.
"""


def is_prime(n: int) -> bool:
    """Check if a number is prime."""
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True


def prime_factorization(n: int) -> list[int]:
    """Return the prime factorization of n."""
    factors = []
    d = 2
    while d * d <= n:
        while (n % d) == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def primitive_root(p: int) -> int:
    """Return a primitive root of p."""
    if not is_prime(p):
        raise ValueError("Input must be a prime number")
    if p == 2:
        return 1
    phi = p - 1
    factors = prime_factorization(phi)
    for g in range(2, p):
        if all(pow(g, phi // factor, p) != 1 for factor in factors):
            return g
    raise ValueError("No primitive root found")  # pragma: no cover


def extract_roots(c_pow_p: int, p: int, e: int) -> tuple[int, ...]:
    """Return all p-th roots in Z/eZ from a p-th-power residue."""
    base_root = (c_pow_p // p) % e
    step = e // p
    return tuple((base_root + k * step) % e for k in range(p))
