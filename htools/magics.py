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
    @argument('-A', action='store_true', help='Warning mode: always.')
    @argument('-P', action='store_true', help='Boolean flag. If passed, the '
              'change will apply for the rest of the notebook, or until the '
              'user changes it again. The default behavior is to apply the '
              'change only to the current cell.')
    def warn(self, line, cell):
        """Silence warnings for a cell. If the default has been changed to
        hide warnings, the -A flag can be used to show all warnings for a
        cell. The -P flag will make the change persist, at least until the
        user changes it again.
        """
        args = parse_argstring(self.warn, line)
        flags = {flag for flag in hdir(args).keys() if getattr(args, flag)}

        # Default is to ignore warnings. This is because warnings alerts you
        # by default, so the typical use case is to silence warnings.
        if 'A' in flags:
            mode = 'always'
        else:
            mode = 'ignore'

        # Change mode and run cell.
        warnings.filterwarnings(mode)
        get_ipython().run_cell(cell)

        # Reset mode if change is note permanent.
        if 'P' not in flags:
            warnings.resetwarnings()


get_ipython().register_magics(InteractiveMagic, WarningMagic)
