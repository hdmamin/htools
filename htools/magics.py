from IPython.core.interactiveshell import InteractiveShell
from IPython.core.magic import cell_magic, magics_class, Magics


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


get_ipython().register_magics(InteractiveMagic)
