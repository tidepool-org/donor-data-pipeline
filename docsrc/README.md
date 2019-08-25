sphinx
======

## Building the docs
1. Complete the `Project Setup` from the Readme.mb that is located at the root of the donor-docs-pipeline project. 

2. To build the content
    - Remove existing files : from within the donor-docs-pipeline/docsrc directory run `make clean`. The donor-docs-pipeline/docs
    directory should now be empty. 
    - Build the new docs : from within the donor-docs-pipeline/docsrc directory run `make html`. The donor-docs-pipeline/docs directory should now have the new content. 

3. Test the build by opening the donor-docs-pipeline/docs/index.html page. 
 
## Adding additional modules
1. Add documentation to the source files within the module according https://www.sphinx-doc.org/en/master/contents.html
2. Copy an existing or create a new .rst file, name it the new modules name and save 
it within donor-docs-pipeline/docs/docsrc. 

3. Add the name of the module in the index.rst. The placement in the file will specify the order
on the sites front page. 

4. Follow the steps in `Building the docs` to view the resulting html.

  