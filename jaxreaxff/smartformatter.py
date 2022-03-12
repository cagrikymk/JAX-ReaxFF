import argparse
# from: https://stackoverflow.com/questions/3853722/how-to-insert-newlines-on-argparse-help-text
class SmartFormatter(argparse.ArgumentDefaultsHelpFormatter):

    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()  
        return argparse.ArgumentDefaultsHelpFormatter._split_lines(self, text, width)
