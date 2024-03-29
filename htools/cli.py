import ast
from datetime import datetime
import fire
from functools import wraps
import json
import pandas as pd
from pathlib import Path
import pkg_resources as pkg
from pkg_resources import DistributionNotFound
import pyperclip
import subprocess
import sys
import warnings

from htools.core import tolist, flatten, save
from htools.meta import get_module_docstring, in_standard_library, source_code


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
# TODO: adjust readme updater init so we can pass in lib-dirs and non-lib dirs
# separately. Ran into some annoying issues where setup.py in parent doesn't
# actually mean a dir is a package.
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


def _pypi_safe_name(name):
    """Try to map from import name to install name. Going the other direction
    is straightforward using pkg_resources but this is harder.

    Parameters
    ----------
    name: str
        Import name of library, e.g. sklearn, requests.

    Returns
    -------
    str: Install name of library. This is NOT foolproof: we make a few
    hard-coded replacements for popular libraries (sklearn -> scikit-learn)
    and replace underscores with dashes, but there's no guarantee this will
    catch everything.
    """
    # Common packages where install name differs from import name. It's easy to
    # go from install name to import name but harder to go the reverse
    # direction.
    import2pypi = {
        'bs4': 'beautifulsoup4',
        'sklearn': 'scikit-learn',
        'PIL': 'pillow',
        'yaml': 'pyyaml',
    }
    return import2pypi.get(name, name.replace('_', '-'))


def module_dependencies(path, package='', exclude_std_lib=True):
    """Find a python script's dependencies. Assumes a relatively standard
    import structure (e.g. no programmatic imports using importlib).

    # TODO: at the moment, this does not support:
    - relative imports
    - libraries whose install name differs from its import name aside from
    popular cases like scikit-learn (in other cases, this would return the
    equivalent of "sklearn")

    Parameters
    ----------
    path: str or Path
        Path to the python file in question.
    package: str
        If provided, this should be the name of the library the module belongs
        to. This will help us differentiate between internal and external
        dependencies. Make sure to provide this if you want internal
        dependencies.
    exclude_std_lib: bool
        Since we often use this to help generate requirements files, we don't
        always care about built-in libraries.

    Returns
    -------
    tuple[list]: First item contains external dependencies (e.g. torch). Second
    item contains internal dependencies (e.g. htools.cli depends on htools.core
    in the sense that it imports it).
    """
    # `skip` is for packages that are imported and aren't on pypi but don't
    # show up as being part of the standard library (in the case of pkg_
    # resources, I believe it's part of a library that IS part of the standard
    # library but from the way it's imported that's not clear to our parser.
    # These MUST be import names (not install names) if they differ.
    skip = {'pkg_resources'}
    with open(path, 'r') as f:
        tree = ast.parse(f.read())
    libs = []
    internal_modules = []
    for obj in tree.body:
        if isinstance(obj, ast.ImportFrom):
            parts = obj.module.split('.')
            if parts[0] == package:
                if len(parts) > 1:
                    internal_modules.append('.'.join(parts[1:]))
                else:
                    assert len(obj.names) == 1, \
                        'Your import seems to have multiple aliases, which ' \
                        'we don\'t know how to process.'
                    assert isinstance(obj.names[0], ast.alias), \
                        f'Expected object name to be an alias but it ' \
                        f'was {obj.names[0]}.'
                    internal_modules.append(obj.names[0].name)
            else:
                libs.append(obj.module)
        elif isinstance(obj,  ast.Import):
            names = [name.name for name in obj.names]
            assert len(names) == 1, f'Error parsing import: {names}.'
            libs.append(names[0])
    # Make sure to filter out `skip` before applying _pypi_safe_name.
    libs = set(_pypi_safe_name(lib.partition('.')[0]) for lib in libs
               if lib not in skip)
    if exclude_std_lib:
        libs = (lib for lib in libs if not in_standard_library(lib))
    return sorted(libs), sorted(internal_modules)


def _resolve_dependencies(mod2ext, mod2int):
    """Fully resolve dependencies: if module "a" depends on "b" in the same
    package (an "internal" dependency), "a" implicitly depends on all of "b"'s
    external dependencies.

    Parameters
    ----------
    mod2ext: dict[str, list]
        Maps module name to list of names of external dependencies (e.g.
        torch).
    mod2int: dict[str, list]
        Maps module name to list of names of internal dependencies.

    Returns
    -------
    dict[str, list]: Maps module name to list of module names (external only,
    but accounts for implicit dependencies).
    """
    old = {}
    new = {k: set(v) for k, v in mod2ext.items()}
    # If module a depends on b and b depends on c, we may require
    # multiple rounds of updates.
    while True:
        for k, v in mod2ext.items():
            new[k].update(flatten(new[mod] for mod in mod2int[k]))
        if old == new: break
        old = new
    return {k: sorted(v) for k, v in new.items()}


def library_dependencies(lib, skip_init=True):
    """Find libraries a library depends on. This helps us generate
    requirements.txt files for user-built packages. It also makes it easy to
    create different dependency groups for setup.py, allowing us to install
    htools[meta] or htools[core] (for example) instead of all htools
    requirements if we only want to use certain modules.

    Runs in the current working directory.

    # TODO: at the moment, this does not support:
    - relative imports
    - nested packages
    - running from different directories (currently it just checks all python
        files in the current directory)
    - libraries whose install name differs from its import name except for
    popular cases like scikit-learn (in other cases, this would return the
    equivalent of "sklearn")
    - imports like "from library import module as alias"

    Parameters
    ----------
    lib: str
        Name of library.
    skip_init: bool
        If True, ignore the __init__.py file.

    Returns
    -------
    dict: First item is a list of all dependencies. Second item is a dict
    mapping module name to a list of its external dependencies. Third is a dict
    mapping module name to a list of its internal dependencies. Fourth is a
    dict mapping module name to a fully resolved list of external dependencies
    (including implicit dependencies: e.g. if htools.core imports requests and
    htools.meta imports htools.core, then htools.meta depends on requests too).
    """
    mod2deps = {}
    mod2int_deps = {}
    for path in Path('.').iterdir():
        if path.suffix != '.py' or (skip_init and path.name == '__init__.py'):
            continue
        try:
            external, internal = module_dependencies(path, lib)
        except AssertionError as e:
            raise RuntimeError(f'Error processing {path.name}: {e}')
        mod2deps[path.stem] = external
        mod2int_deps[path.stem] = internal
    fully_resolved = _resolve_dependencies(mod2deps, mod2int_deps)
    all_deps = set(sum(mod2deps.values(), []))
    return dict(overall=sorted(all_deps),
                external=mod2deps,
                internal=mod2int_deps,
                resolved=fully_resolved)


def _libs2readme_str(lib2version):
    return '\n'.join(f'{k}=={v}' if v else k for k, v in lib2version.items())


# TODO: figure out how to handle __init__.py when finding deps (maybe need to
# wrap each star import in try/except?). Also add extra func so the CLI command
# for find_dependencies generates text/markdown/yaml/json files we can load in
# setup.py.
def make_requirements_file(lib, skip_init=True, make_resolved=False,
                           out_path='../requirements.txt'):
    """Generate a requirements.txt file for a project by extracting imports
    from the python source code. You must run this from the lib directory, e.g.
    from something like ~/project/lib/htools.

    Parameters
    ----------
    lib: str
    skip_init: bool
    make_resolved: bool
    out_path: str

    Returns
    -------
    str
    """
    deps = library_dependencies(lib, skip_init)
    lib2version = {}
    for lib in deps['overall']:
        try:
            lib2version[lib] = pkg.get_distribution(lib).version
        except DistributionNotFound:
            warnings.warn(
                f'Could not find {lib} installed. You should confirm '
                'if its pypi name differs from its import name.'
            )
            lib2version[lib] = None

    # Need to sort again because different between import name and install name
    # can mess up our ordering.
    file_str = _libs2readme_str(lib2version)
    save(file_str, out_path)

    # If desired, generate a json mapping each module to its own requirements
    # file. Modules with no dependencies will not have a key. The json file
    # can then be loaded in setup.py to easily create a number of different
    # `install_requires` variants.
    if make_resolved:
        out_dir = Path(out_path).parent
        module2readme = {}
        for mod, libs in deps['resolved'].items():
            readme = _libs2readme_str({lib: lib2version[lib] for lib in libs})
            if readme: module2readme[mod] = readme
        save(module2readme, out_dir/'module2readme.json')

    return file_str


# TODO: might be cleaner to compile all this readme functionality into a single
# class.
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


def source(name, lib='htools', copy=False):
    """Print or copy the source code of a class/function defined in htools.

    Parameters
    ----------
    name: str
        Class or function defined in htools.
    lib: str
        Name of library containing `name`, usually 'htools'. Won't work on
        the standard library or large complex libraries (specifically, those
        with nested file structures).
    copy: bool
        If True, copy the source code the clipboard. If False, simply print it
        out.

    Returns
    -------
    str: Source code of htools class/function.

    Examples
    --------
    # Copies source code of auto_repr decorator to clipboard. Excluding the
    # -c flag will simply print out the source code.
    htools src auto_repr -c
    """
    src, backup_name = source_code(name, lib_name=lib)
    if not src:
        print(f'Failed to retrieve `{name}` source code from {lib}.')
        if backup_name != name:
            cmd = f'{lib} src {backup_name}'
            if lib != 'htools':
                cmd += f' --lib = {lib}'
            print(f'We suggest trying the command:\n\n{cmd}')
    if copy:
        pyperclip.copy(src)
    else:
        print(src)


def cli():
    fire.Fire({
        'update_readmes': update_readmes,
        'make_requirements': make_requirements_file,
        'src': source
    })

