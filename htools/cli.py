import ast
from datetime import datetime
import fire
import json
import pandas as pd
from pathlib import Path
import subprocess

from htools.core import tolist


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
    """This is generally intended for a structure where each directory contains
    either python scripts OR jupyter notebooks - I haven't tested it on
    directories containing both. This is also not intended to work recursively:
    we only try to update 1 readme file per directory.
    """

    time_fmt = '%Y-%m-%d %H:%M:%S'
    readme_id_start = '\n---\nStart of auto-generated file data.<br/>'
    readme_id_end = '\n<br/>End of auto-generated file data. Do not add ' \
                    'anything below this.\n'
    readme_regex = readme_id_start + '(.|\n)*' + readme_id_end
    last_edited_cmd_fmt = 'git log -1 --pretty="format:%ct" {}'

    def __init__(self, *dirs, default='_'):
        """
        Parameters
        ----------
        dirs: str
            One or more paths (we recommend entering these relative to the
            project root, where you should be running the command from, though
            absolute paths should work too).
        default: str
            Used when a python file lacks a module-level docstring or a jupyter
            notebook lacks a "# Summary" markdown cell near the top.

        Example
        -------
        updater = ReadmeUpdater('lib', 'notebooks')
        updater.update_dirs()
        """
        self.dirs = [Path(d) for d in dirs]
        self.extensions = {'.py', '.ipynb'}
        self.default = default

    def update_dirs(self, *dirs):
        """Update README files in the relevant directories.

        Parameters
        ----------
        dirs: str
            If none are provided, this defaults to `self.dirs`. If you specify
            values, this will process only those directories.
        """
        for dir_ in dirs or self.dirs:
            file_df = self._parse_dir_files(dir_)
            self.update_readme(dir_/'README.md', file_df)

    def _parse_dir_files(self, dir_):
        files = []
        for path in dir_.iterdir():
            if path.suffix not in self.extensions: continue
            stats = path.stat()
            files.append({
                'File': path.parts[-1],
                'Summary': self.parse_file(path) or self.default,
                'Last Modified': self.last_modified_date(path),
                'Size': self.readable_file_size(stats.st_size)
            })
        return pd.DataFrame(files).sort_values('File')

    def parse_file(self, path):
        return getattr(self, f'_parse_{path.suffix[1:]}')(path)

    def update_readme(self, path, file_df):
        path.touch()
        with open(path, 'r+') as f:
            text = f.read().split(self.readme_id_start)[0] \
                   + self._autogenerate_text(file_df)
            f.seek(0)
            f.write(text)

    def _autogenerate_text(self, df):
        date_str = 'Last updated: ' + datetime.now().strftime(self.time_fmt)
        autogen = (self.readme_id_start + date_str + '\n\n'
                   + df.to_html(index=False).replace('\\n', '<br/>')
                   + self.readme_id_end)
        return autogen

    def _parse_py(self, path):
        with open(path, 'r') as f:
            tree = ast.parse(f.read())
        return ast.get_docstring(tree)

    def _parse_ipynb(self, path):
        with open(path, 'r') as f:
            cells = json.load(f)['cells'][:3]
        for cell in cells:
            if cell['cell_type'] == 'markdown' and \
                    'summary' in cell['source'][0].lower():
                # Notebook lines include newlines so we don't add them back in.
                return ''.join(cell['source'][1:]).strip()
        return ''

    def timestamp_to_time_str(self, time):
        return datetime.fromtimestamp(time).strftime(self.time_fmt)

    def last_modified_date(self, path):
        try:
            # If we're in a git repo, file edit times are changed when we pull
            # so we have to use built-in git functionality. This will fail if
            # we call the command from a different repo. I vaguely recall
            # seeing weird git behavior inside running docker containers so I'm
            # not sure if this will work there.
            git_time = subprocess.check_output(
                self.last_edited_cmd_fmt.format(path).split()
            )
            timestamp = int(git_time.decode().strip()
                                    .replace('format:', '').replace('"', ''))
        except Exception as e:
            timestamp = path.stat().st_ctime
        return self.timestamp_to_time_str(timestamp)

    @staticmethod
    def readable_file_size(n_bytes):
        power = len(str(n_bytes)) - 1
        assert power < 24, 'Are you sure file is larger than a zettabyte?'

        prefix_powers =[
            (0, 'b'),
            (3, 'kb'),
            (6, 'mb'),
            (9, 'gb'),
            (12, 'tb'),
            (15, 'pb'),
            (18, 'eb'),
            (21, 'zb'),
            (24, 'yb')
        ]
        prev_pow = 0
        prev_pre = 'b'
        for curr_pow, curr_pre in prefix_powers:
            if power < curr_pow: break
            prev_pow = curr_pow
            prev_pre = curr_pre
        return f'{(n_bytes / 10**prev_pow):.2f} {prev_pre}'


def update_readmes(dirs, default='_'):
    """Update readme files with a table of info about each python file or ipy
    notebook in the relevant directory. This relies on python files having
    module level docstrings and ipy notebooks having a markdown cell starting
    with '# Summary' (this must be one of the first 5 cells in the notebook).
    We also provide info on last edited times and file sizes.

    Parameters
    ----------
    dirs: str or list[str]
        One or more directories to update readme files for.
    default: str
        Default value to use when no docstring/summary is available.
    """
    parser = ReadmeUpdater(*tolist(dirs), default=default)
    parser.update_dirs()


def cli():
    fire.Fire({
        'update_readmes': update_readmes
    })

