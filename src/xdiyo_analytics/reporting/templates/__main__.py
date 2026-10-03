"""python -m xdiyo_analytics.reporting.templates tickets|curves|portfolio MANIFEST -o DIR"""
import argparse
from . import build_tickets, build_curves, build_portfolio


def main():
    parser = argparse.ArgumentParser(description='Render exported evidence offline; no model or experiment execution.')
    parser.add_argument('format', choices=['tickets', 'curves', 'portfolio'])
    parser.add_argument('manifest')
    parser.add_argument('-o', '--output', required=True, help='Empty output directory')
    args = parser.parse_args()
    builder = {'tickets': build_tickets, 'curves': build_curves, 'portfolio': build_portfolio}[args.format]
    try:
        print(builder(args.manifest, args.output))
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f'Cannot build report: {error}\n')


if __name__ == '__main__':
    main()
