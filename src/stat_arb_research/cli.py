from __future__ import annotations

import argparse

from stat_arb_research.figures import create_all_figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Statistical arbitrage research utilities")
    parser.add_argument(
        "--figures",
        action="store_true",
        help="Regenerate the report figures under reports/figures.",
    )
    args = parser.parse_args()

    if args.figures:
        metrics = create_all_figures()
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
