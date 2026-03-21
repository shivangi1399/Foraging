import sys

MOD = 10**9 + 7


def ans(N, A, B):
    # Sort 1-blocks descending: put largest first to maximize high-order 1s
    A.sort(reverse=True)
    # Sort 0-blocks ascending: put shortest first to minimize gaps between 1-blocks
    B.sort()

    # Total string length
    L = sum(A) + sum(B)

    result = 0
    remaining = L  # bits remaining to the right of current position (exclusive)

    for i in range(N):
        # Place A[i] ones
        a = A[i]
        remaining -= a
        # Contribution: 111...1 (a bits) at this position
        # = (2^a - 1) * 2^remaining
        contribution = (pow(2, a, MOD) - 1) * pow(2, remaining, MOD) % MOD
        result = (result + contribution) % MOD

        # Place B[i] zeros (no contribution to value)
        remaining -= B[i]

    return result


def main():
    data = sys.stdin.read().split()
    idx = 0
    N = int(data[idx]); idx += 1
    A = []
    for _ in range(N):
        A.append(int(data[idx])); idx += 1
    B = []
    for _ in range(N):
        B.append(int(data[idx])); idx += 1

    result = ans(N, A, B)
    print(result)


if __name__ == '__main__':
    main()
