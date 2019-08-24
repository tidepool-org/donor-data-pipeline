sphinx
======

## Getting started - building from source.
1. Follow the `Project Setup` from the Readme.mp from the root of the project. 

2. To build the content run `build clean` to remove existing files then run `build gethub' to generate the 
pages which will be saved in the donar-docs-pipeline/docs folder. 

3. Test the build by opening the donar-docs-pipeline/docs/index.html page. 
 
## Adding additional modules
1. Add documentation to the source files within the module according https://www.sphinx-doc.org/en/master/contents.html
2. Copy an existing or create a new .rst file, name it the new modules name and save 
it within donar-docs-pipeline/docs/docsrc. 

3. Add the name of the module in the index.rst. The placement in the file will specify the order
on the sites front page. 

4. Follow the steps in `Getting started - building from source` to view the resulting html.

  