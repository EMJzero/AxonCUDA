import argparse
import random
import sys

# sloppy code, courtesy of ChatGPT :)

def parse_hmetis(filename):
    """
    Reads an hMETIS hypergraph file.
    Returns:
        E, V, fmt, hyperedges
    where hyperedges is a list of tuples:
        (weight_or_None, [v1, v2, ...])
    """
    hyperedges = []
    fmt = 0

    with open(filename, "r") as f:
        # skip comments until header
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            header = list(map(int, line.split()))
            if len(header) == 2:
                E, V = header
                fmt = 0
            elif len(header) == 3:
                E, V, fmt = header
            else:
                raise ValueError("Invalid hMETIS header")
            break

        # read hyperedges
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            parts = list(map(int, line.split()))
            if fmt in (1, 11):
                w = parts[0]
                vs = parts[1:]
            else:
                w = None
                vs = parts
            hyperedges.append((w, vs))

    if len(hyperedges) != E:
        raise ValueError(f"Expected {E} hyperedges, found {len(hyperedges)}")

    return E, V, fmt, hyperedges


def write_hmetis(filename, V, fmt, hyperedges):
    with open(filename, "w") as f:
        if fmt == 0:
            f.write(f"{len(hyperedges)} {V}\n")
        else:
            f.write(f"{len(hyperedges)} {V} {fmt}\n")

        for w, vs in hyperedges:
            if w is not None:
                f.write(str(w) + " ")
            f.write(" ".join(map(str, vs)) + "\n")


def scale_hypergraph(E, V, fmt, hyperedges, factor):
    target_E = int(E * factor)
    target_V = int(V * factor)

    next_vertex_id = V + 1
    all_hyperedges = list(hyperedges)

    # probability of reusing an existing vertex instead of a new one
    reuse_prob = 0.2

    while len(all_hyperedges) < target_E:
        w, vs = random.choice(hyperedges)
        k = len(vs)

        new_vs = []
        for _ in range(k):
            if random.random() < reuse_prob:
                new_vs.append(random.randint(1, next_vertex_id - 1))
            else:
                if next_vertex_id <= target_V:
                    new_vs.append(next_vertex_id)
                    next_vertex_id += 1
                else:
                    new_vs.append(random.randint(1, target_V))

        # remove duplicates inside hyperedge
        new_vs = list(sorted(set(new_vs)))

        # avoid degenerate edges
        if len(new_vs) >= 2:
            all_hyperedges.append((w, new_vs))

    return target_V, all_hyperedges


def main():
    parser = argparse.ArgumentParser(description="Scale an hMETIS hypergraph by adding random nodes and hyperedges.")
    parser.add_argument("input_file", help="Input hMETIS file")
    parser.add_argument("output_file", help="Output hMETIS file")
    parser.add_argument("factor", type=float, help="Multiplicative scaling factor (e.g., 10)")

    args = parser.parse_args()

    if args.factor <= 1.0:
        sys.exit("Factor must be > 1")

    E, V, fmt, hyperedges = parse_hmetis(args.input_file)
    new_V, new_hyperedges = scale_hypergraph(E, V, fmt, hyperedges, args.factor)
    write_hmetis(args.output_file, new_V, fmt, new_hyperedges)


if __name__ == "__main__":
    main()