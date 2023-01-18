# ecommerce_unsupervised_classifier
<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">

  <h3 align="center">Unsupervised Ecommerce Customer Categorization</h3>

  <p align="center">
    Python implementation to try to group similar customers of an ecommerce website
    <br />
    <a href="https://github.com/ajbrumleve/ecommerce_unsupervised_classifier/issues">Report Bug</a>
    ·
    <a href="https://github.com/ajbrumleve/ecommerce_unsupervised_classifier">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project


This was my first attempt to really try to build an unsupervised model to try to solve a real world problem. This uses a [Kaggle dataset](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store?select=2019-Oct.csv) of customer interactions with a multi category online store. I wanted to see if I could create groupings of customers and then be able to place a customer in a grouping based on a shopping session. Applications of this would be to recommend tailored products to each customer.

When a new model is trained, a vector is created to represent the categories viewed and purchased by each user. The code automatically uses the elbow method to identify how many groups to use for the KMeans model. A model is then created with that number of groups. Next a separate model is trained which looks at each session and which categories are viewed or purchased as well as the customer vector at the time of the session. Using the customer grouping from the original model, a Support Vector Machine model is trained to put sessions into one of the customer groupings. 

The model is saved in the models directory so it can be accessed in the future without needing to retrain. 

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

This section should list any major frameworks/libraries used to bootstrap your project. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples.

* [![Python][Python]][Python-url]


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started



### Prerequisites

Require packages kneed, scikit-learn, and numpy.

Use `pip` to install the packages from PyPI:

```bash
pip install kneed
pip install scikit-learn
pip install numpy
```


### Installation



1. Download and unzip [this entire repository from GitHub](https://github.com/ajbrumleve/ecommerce_unsupervised_classifier), either interactively, or by entering the following in your Terminal.
    ```bash
    git clone https://github.com/ajbrumleve/ecommerce_unsupervised_classifier.git
    ```
2. Navigate into the top directory of the repo on your machine
    ```bash
    cd ecommerce_customer_categorization
    ```
3. Create a virtualenv and install the package dependencies. If you don't have `pipenv`, you can follow instructions [here](https://pipenv.pypa.io/en/latest/install/) for how to install.
    ```bash
    pipenv install
    ```
4. Run `main.py` to try to try prebuilt model. Or run `march_madness.py` to train new model. This should be fairly quick with the sample data. With the full data from Kaggle, this can take a while. 


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

With a trained model, running main.py will ask you for the customer_id of a customer session. Then you will list the categories the user views or purchases. If the session ends enter the word 'Finally'. The program will then tell you which customer category the customer is in.

<p align="right">(<a href="#readme-top">back to top</a>)</p>




<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

Some ideas of ways to extend this code include:
 - Adding code to choose the top 3 products to recommend to a given user.
 - Adding the ability to categorize each session individually
 - Adding new features to the model. Right now the model does not factor in brands or specific items viewed. It only looks at categories. 

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Andrew Brumleve - [@AndrewBrumleve](https://twitter.com/AndrewBrumleve) - ajbrumleve@gmail.com

Project Link: [https://github.com/

ecommerce_unsupervised_classifier](https://github.com/ecommerce_unsupervised_classifier)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [Kaggle](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store)
* [README Template](https://github.com/othneildrew/Best-README-Template)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/ajbrumleve/ecommerce_unsupervised_classifier.svg?style=for-the-badge
[contributors-url]: https://github.com/ajbrumleve/ecommerce_unsupervised_classifier/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/ajbrumleve/ecommerce_unsupervised_classifier.svg?style=for-the-badge
[forks-url]: https://github.com/ajbrumleve/ecommerce_unsupervised_classifier/network/members
[stars-shield]: https://img.shields.io/github/stars/ajbrumleve/ecommerce_unsupervised_classifier.svg?style=for-the-badge
[stars-url]: https://github.com/ajbrumleve/ecommerce_unsupervised_classifier/stargazers
[issues-shield]: https://img.shields.io/github/issues/ajbrumleve/ecommerce_unsupervised_classifier.svg?style=for-the-badge
[issues-url]: https://github.com/ajbrumleve/ecommerce_unsupervised_classifier/issues
[license-shield]: https://img.shields.io/github/license/ajbrumleve/ecommerce_unsupervised_classifier.svg?style=for-the-badge
[license-url]: https://github.com/ajbrumleve/ecommerce_unsupervised_classifier/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: (https://www.linkedin.com/in/andrew-brumleve-574239227/)
[product-screenshot]: images/screenshot.png
[Python]:  	https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://python.org/
