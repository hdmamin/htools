import fire
from pathlib import Path
import re

from htools.core import save, load


def Display(lines, out):
    """Monkeypatch Fire CLI to print "help" to stdout instead of using `less`
    window. User never calls this with arguments so don't worry about them.

    import fire

    def main():
        # do something

    if __name__ == '__main__':
        fire.core.Display = Display
        fire.Fire(main)
    """
    out.write('\n'.join(lines) + '\n')


fire.core.Display = Display


# Start of htools CLI below. Stuff above is to import when building new CLIs
# in other projects.
class ReadmeUpdater:

    def __init__(self, dirs):
        self.dirs = [Path(d) for d in dirs]
        self.extensions = {'.py', '.ipynb'}
        self.docstring_chars = ('"""', "'''")
        self.docstring_regex = '^(\'\'\'|\"\"\")Summary(.|\n)*(\'\'\'|\"\"\")$'

    def process_files(self):
        for dir_ in self.dirs:
            self._process_dir_files(dir_)

    def _process_dir_files(self, dir_):
        self.path2summary = {path: self.parse_file() for path in dir_.iterdir()
                             if path.suffix in self.extensions}

    def parse_file(self, path):
        return getattr(self, f'_parse_{path.suffix}')(path)

    def _parse_py(self, path):
        with open(path, 'r') as f:
            text = f.read()
        match = re.search(self.docstring_regex, text, re.MULTILINE)
        # for i, line in enumerate(text.splitlines[:5]):
        #     if line.startswith():
        #         break


    def _parse_ipynb(self, path):
        pass


def _update_readme():
    pass


def update_readme():
    pass


if __name__ == '__main__':
    fire.Fire({
        'update_readme': update_readme
    })
