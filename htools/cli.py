import ast
from datetime import datetime
import fire
from functools import wraps
import json
import pandas as pd
from pathlib import Path
import subprocess
import sys

from htools.core import tolist
from htools.meta import get_module_docstring


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

    def __init__(self, *dirs, default='_', detect_library=True):
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
        detect_library: bool
            If True, we try to check if each directory is a python library. If
            it is, the readme file will be placed in its parent rather than the
            dir with all the python files. This is useful if you have py files
            in `lib/my_library_name` but want to update the readme in `lib`.

        Example
        -------
        updater = ReadmeUpdater('bin', 'lib/my_library_name', 'notebooks')
        updater.update_dirs()
        """
        self.dirs = [Path(d) for d in dirs]
        self.extensions = {'.py', '.ipynb'}
        self.default = default
        self.detect_library = detect_library

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
            if file_df.empty: continue
            # In this first scenario, we check files in the specified path but
            # update the readme of its parent. This is useful for a file
            # structure like `lib/my_library_name`: we want to parse files from
            # `lib/my_library_name` but update the readme in `lib`.
            if self.detect_library and 'setup.py' in \
                    set(p.parts[-1] for p in dir_.parent.iterdir()):
                readme_path = dir_.parent/'README.md'
            else:
                readme_path = dir_/'README.md'
            self.update_readme(readme_path, file_df)

    def _parse_dir_files(self, dir_):
        """Extract information (summary, modify time, size, etc.) from each
        python script or ipy notebook in a directory.

        Parameters
        ----------
        dir_: Path
            Directory to parse. The intention is this should either contain
            notebooks OR python scripts since that's my convention - I'm not
            sure if it will work otherwise.

        Returns
        -------
        pd.DataFrame: 1 row for each relevant file. This works best with my
        convention of naming files like "nb01_eda.ipynb" or
        "s01_download_data.py" since results are sorted by name.
        """
        files = []
        for path in dir_.iterdir():
            if path.suffix not in self.extensions: continue
            stats = path.stat()
            # Want py/ipy custom fields to come before change time/size in df.
            files.append({
                'File': path.parts[-1],
                **self.parse_file(path),
                'Last Modified': self.last_modified_date(path),
                'Size': self.readable_file_size(stats.st_size)
            })

        # File numbering convention means these should be displayed in a
        # logical order. Sort columns so name and summary are first.
        df = pd.DataFrame(files)
        if df.empty: return df
        return df.sort_values('File').reset_index(drop=True)

    def parse_file(self, path):
        """Wrapper to parse a python script or ipy notebook.

        Parameters
        ----------
        path

        Returns
        -------
        dict: Should have key 'Summary' regardless of file type. Other keys
        vary depending on whether the file is a script or notebook.
        """
        return getattr(self, f'_parse_{path.suffix[1:]}')(path)

    def update_readme(self, path, file_df):
        """Load a readme file, replace the old auto-generated table with the
        new one (or adds it if none is present), and writes back to the same
        path.

        Parameters
        ----------
        path: Path
            Readme file location. Will be created if it doesn't exist.
        file_df: pd.DataFrame
            1 row for each file, where columns are things like file
            name/size/summary.
        """
        path.touch()
        with open(path, 'r+') as f:
            text = f.read().split(self.readme_id_start)[0] \
                   + self._autogenerate_text(file_df)
            f.seek(0)
            f.write(text)

    def _autogenerate_text(self, df):
        """Create the autogenerated text portion that will be written to a
        readme. This consists of dataframe html (my notebooks/files sometimes
        contain markdown so a markdown table was a bit buggy) sandwiched
        between some text marking the start/end of autogeneration.

        Parameters
        ----------
        df: pd.DataFrame
            DF where 1 row corresponds to 1 file.

        Returns
        -------
        str: Autogenerated text to plug into readme.
        """
        date_str = 'Last updated: ' + datetime.now().strftime(self.time_fmt)
        autogen = (self.readme_id_start + date_str + '\n\n'
                   + df.to_html(index=False).replace('\\n', '<br/>')
                   + self.readme_id_end)
        return autogen

    def _parse_py(self, path):
        """Process a python script to find its summary (module-level docstring)
        and line count.

        Parameters
        ----------
        path: Path

        Returns
        -------
        dict: Information about the file specified by `path`. Additional
        attributes can be added without issue.
        """
        with open(path, 'r') as f:
            text = f.read()
        tree = ast.parse(text)
        return {'Summary': ast.get_docstring(tree) or self.default,
                'Line Count': len(text.splitlines())}

    def _parse_ipynb(self, path):
        """Extract summary and other stats (# of code/markdown cells) from a
        notebook. The summary must be a markdown cell within the first 3 cells
        of the notebook where the first line is '# Summary' (this shows up as a
        large header in markdown).

        Parameters
        ----------
        path: Path

        Returns
        -------
        dict: Information about the file specified by `path`. Additional
        attributes can be added without issue.
        """
        with open(path, 'r') as f:
            cells = json.load(f)['cells']
        res = {
            'Summary': self.default,
            'Code Cell Count': len([c for c in cells
                                    if c['cell_type'] == 'code']),
            'Markdown Cell Count': len([c for c in cells
                                        if c['cell_type'] == 'markdown'])
        }
        for cell in cells[:3]:
            if cell['cell_type'] == 'markdown' and \
                    'summary' in cell['source'][0].lower():
                # Notebook lines include newlines so we don't add them back in.
                res['Summary'] = ''.join(cell['source'][1:]).strip()
                return res
        return res

    def timestamp_to_time_str(self, time):
        """Convert a timestamp to a nicely formatted datetime string.

        Parameters
        ----------
        time: int

        Returns
        -------
        str: Format like '2021/03/31 15:43:00'
        """
        return datetime.fromtimestamp(time).strftime(self.time_fmt)

    def last_modified_date(self, path):
        """Get the last time a file was modified. If in a git repo, this
        information is often uninformative due to pulls, but we try to retrieve
        the data by checking the last commit each file changed in. This will
        fail if running the command from a different repo (which you shouldn't
        really ever do) and fall back to the file system's record of last
        modified time.

        Parameters
        ----------
        path: Path

        Returns
        -------
        str: Date formatted like in `timestamp_to_time_str`.
        """
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
        """Convert a file size in bytes to a human readable unit. Not
        extensively tested but it seems to work so far.

        Parameters
        ----------
        n_bytes: int

        Returns
        -------
        str: File size in a more human readable unit (e.g. mb or gb). A space
        separates the numeric portion and the unit name.
        """
        power = len(str(n_bytes)) - 1
        assert power < 24, 'Are you sure file is larger than a yottabyte?'

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


def module_docstring(func):
    """Decorator to add the current module's docstring to a function's
    docstring. This is intended for use in simple (1 command,
    zero or minimal arguments) fire CLIs where I want to write a single
    docstring for the module and function. Writing it at the module level
    allows htools.cli.ReadmeUpdater to update the appropriate readme, while
    this decorator ensures that the info will be available when using the
    '--help' flag at the command line. Do NOT use this on functions in a
    library - I've only tested it on py scripts and it relies on sys.argv, so
    I'm pretty sure it will break outside of the intended context.
    """
    doc = func.__doc__ or ''
    module_doc = get_module_docstring(sys.argv[0])
    if doc:
        func.__doc__ = module_doc + '\n\n' + doc
    else:
        func.__doc__ = module_doc

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def update_readmes(dirs, default='_'):
    """Update readme files with a table of info about each python file or ipy
    notebook in the relevant directory. This relies on python files having
    module level docstrings and ipy notebooks having a markdown cell starting
    with '# Summary' (this must be one of the first 3 cells in the notebook).
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

