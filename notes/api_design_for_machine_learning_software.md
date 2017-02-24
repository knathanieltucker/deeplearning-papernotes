The people over at scimitar-learn bring together a set of principals and experiences they had from their time working on sklearn. The five principles are:

* Consistency
* Inspection
* Non-proliferation of classes
* Composition
* Sensible defaults

For them, the main idea seems to be ease of use (for the developer and the practitioner). Which is an admirable goal. But for me I gleaned a little be more from it.

The consistency (brought about by the universal data representation of a numpy array) and the non-proliferation of classes force developers into a general API. This does mean ease of use, but also means that development of helper functions (model scoring methods, data preprocessing, internal model helpers) can all be used almost universally. This is a nice thing when developing a new library or new ML company.

And this plays very nicely with composition. To achieve generality, we can make compositional methods (again leveraging the consistent API) that can greatly expand your current capacities. This ability to compose adds another constraint to the mix which is the separation of data and model. You can combine multiple models and jointly train them. Or you can train models and then stitch them together afterwards. 

The paper is not too hard so I will leave the details to you. But I am convinced for most machine learning applications, using sklearn as the base API will help out in the long and short run
