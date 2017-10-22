"""Script to allow code to run via command line."""
import sys

from .embedding import main
main(sys.argv[1:])
