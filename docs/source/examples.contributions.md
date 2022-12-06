# Contributing Examples

Examples complementing the ones provided in the current documentation are always welcome!
In case you have an example and you would like to see it here, feel free to contribute it.
In doing so please make sure to account for the remarks described in the following.

## Preparing the Example

An example should be provided in terms of a jupyter notebook. Besides a descriptive title (and possibly sub-titles) these notebooks should include some explanatory text (```cell type``` is ```Markdown```)
and two types of code blocks (```cell type``` is ```Code```). The first type is used to provide the actual code for the example illustrating the explicit commands used to achieve the operation described in the text.
Code blocks of the second type will not be displayed in the online documentation, but are used for testing purposes only. Consequently, these need to contain some sort of checks (e.g. ```assert``` statements)
that raise an Error in case executing the code of the first type is not working as intended.

## Hiding Tests in the Online Documentation

Before saving the jupyter notebook in ```./docs/source/``` under ```examples.<descriptive_name>.ipynb``` make sure to ```Restart Kernel and Clear All Outputs``` in order to provide it in a defined state.
Afterwards open the file with you favorite text editor, search for all of your code blocks of the second type and add ```"nbsphinx": "hidden"``` to the ```"metadata": {}```.

## Updating Related Files

Update the ```examples.rst``` file by adding a line with your ```examples.<descriptive_name>.ipynb```, as well as the ```examples.index.md``` file by updating the list of common operations with links to your example.

To have the examples actually been tested open the file ```./tests/test_doc_examples.py``` and add a test for the code blocks in your jupyter notebook. In the simplest case this only requires adding a new ```pytest.param('examples.<descriptive_name>.ipynb', 'all', id='<descriptive name>')``` to the ```@pytest.mark.parametrize```.

