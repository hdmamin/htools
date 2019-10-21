from IPython.core.interactiveshell import InteractiveShell
from IPython.core.magic import cell_magic, magics_class, Magics
from IPython.core.magic_arguments import (argument, magic_arguments,
                                          parse_argstring)
import warnings

from htools import hdir


@magics_class
class InteractiveMagic(Magics):

    @cell_magic
    def talk(self, line=None, cell=None):
        """When Jupyter notebook is in default mode where
        ast_node_interactivity=last (i.e. only the last unprinted statement is
        displayed), this will run the current cell while printing all
        statements. It then resets the mode so future cells only print the last
        statement again.

        Examples
        ---------
        In the example below, each cell contains two statements. Notice that
        the cell containing the magic displays both lines of output, while the
        other cells only display the last output.

        >>> 5 + 10
        >>> 6 + 11

        17

        %%talk
        >>> 6 + 2
        >>> 3 + 1

        8
        4

        >>> 1 + 2
        >>> 3 + 4

        7
        """
        InteractiveShell.ast_node_interactivity = 'all'
        get_ipython().run_cell(cell)
        InteractiveShell.ast_node_interactivity = 'last'

    @cell_magic
    def hush(self, line=None, cell=None):
        """The reverse of the `talk` magic. When the notebook is in
        ast_node_interactivty='all' mode, this can be used to suppress outputs
        other than the last one for a single cell. Cells that follow will
        return to the display mode set for the whole notebook.

        Examples
        ---------
        In the example below, each cell contains two statements. Notice that
        the cell containing the magic only displays the last line of output,
        while the other cells display both outputs.

        >>> 5 + 10
        >>> 6 + 11

        15
        17

        %%hush
        >>> 6 + 2
        >>> 3 + 1

        4

        >>> 1 + 2
        >>> 3 + 4

        3
        7
        """
        InteractiveShell.ast_node_interactivity = 'last'
        get_ipython().run_cell(cell)
        InteractiveShell.ast_node_interactivity = 'all'

    @cell_magic
    def mute(self, line=None, cell=None):
        """A more extreme version of the `hush` magic that suppresses all
        output from a cell. Cells that follow will return to the default mode
        of ast_node_interactivity='last'.

        Examples
        ---------
        In the example below, each cell contains two statements. Notice that
        the cell containing the magic displays no output, while the other cells
        display the final output.

        >>> 5 + 10
        >>> 6 + 11

        17

        %%mute
        >>> 6 + 2
        >>> 3 + 1


        >>> 1 + 2
        >>> 3 + 4

        7
        """
        InteractiveShell.ast_node_interactivity = 'none'
        get_ipython().run_cell(cell)
        InteractiveShell.ast_node_interactivity = 'all'


@magics_class
class WarningMagic(Magics):

    @cell_magic
    @magic_arguments()
    @argument('-p', action='store_true', help='Boolean flag. If passed, the '
              'change will apply for the rest of the notebook, or until the '
              'user changes it again. The default behavior is to apply the '
              'change only to the current cell.')
    def lax(self, line, cell):
        """Silence warnings for a cell. The -p flag can be used to make the
        change persist, at least until the user changes it again.
        """
        args = parse_argstring(self.lax, line)
        self._warn(cell, 'ignore', args.p)

    @cell_magic
    @magic_arguments()
    @argument('-p', action='store_true', help='Boolean flag. If passed, the '
              'change will apply for the rest of the notebook, or until the '
              'user changes it again. The default behavior is to apply the '
              'change only to the current cell.')
    def nag(self, line, cell):
        """Silence warnings for a cell. The -p flag can be used to make the
        change persist, at least until the user changes it again.
        """
        args = parse_argstring(self.nag, line)
        self._warn(cell, 'always', args.p)

    def _warn(self, cell, mode, persist):
        """Base method for lax and nag. These could easily be handled in a
        single method with optional flags, but I find the usage to be more
        intuitive when the names are different, and generally prefer flag-free
        magics since the goal is ease of use.

        The persist flag is processed in the child methods because parsing
        references the method that was called.
        """
        warnings.filterwarnings(mode)
        get_ipython().run_cell(cell)

        # Reset manually because warnings.resetwarnings() behaved erratically.
        if not persist:
            out_modes = {'ignore', 'always'}
            out_modes.remove(mode)
            warnings.filterwarnings(list(out_modes)[0])


@magics_class
class FunctionRacerMagic(Magics):

    @cell_magic
    @magic_arguments()
    @argument('-n', help='Number of loops when timing functions (inner loop).')
    @argument('-r', help='Number of runs when timing functions (outer loop).')
    def race(self, line, cell):
        """Time 2 or more functions to allow the user to easily compare speeds.
        Each line will be timed separately, so a function call cannot take up
        multiple lines. This is essentially a convenient wrapper for the
        %%timeit magic that ensures all functions are timed with the same
        choice of parameters. (When timing each function separately, I found
        that during the testing process I would often end up changing some
        function or timeit parameters in one case but forget to change it for
        another. This magic aims to prevent that situation.)

        Examples
        ---------
        Example 1: A fairly standard case where we time three possible
        implementations of a function to see which is fastest.

        %%race -n 10 -r 3
        >>> tokenizer_v1(text)
        >>> tokenizer_v2(text)
        >>> tokenizer_v3(text)

        Example 2: If a function requires many arguments or if parameter
        names are long, consider passing in a list or dictionary of arguments.

        %%race
        >>> many_args_func_v1(**params)
        >>> many_args_func_v2(**params)
        """
        args = parse_argstring(self.race, line)
        n = args.n or 5
        r = args.r or 3

        # Split cell into lines of code to execute.
        rows = [row for row in cell.strip().split('\n')
                if not row.startswith('#')]
        prefix = f'%timeit -n {n} -r {r} '
        for row in rows:
            get_ipython().run_cell(prefix + row)


get_ipython().register_magics(InteractiveMagic, WarningMagic,
                              FunctionRacerMagic)
