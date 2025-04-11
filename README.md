
<h1 style="border-bottom: 2px solid #4CAF50; padding-bottom: 10px;">
  Hyperspectral Imaging for Blood Stain Identification in Forensic Science
</h1>

<a href="https://ieeexplore.ieee.org/abstract/document/10493757" target="_blank"> IEEE Publication Link</a>
<br>

<p style="text-align: justify; font-family: Arial, sans-serif;">
  This project highlights the potential of hyperspectral imaging as a non-invasive technique for identifying and classifying blood stains in forensic science, eliminating the need for physical sampling of critical evidence. Traditional chemical processes used for blood identification and classification often compromise subsequent DNA analysis, necessitating the exploration of innovative methods. However, developing robust algorithms for blood detection using hyperspectral imaging is challenging due to the high dimensionality of the data and the limited availability of training samples.
</p>

<p style="text-align: justify; font-family: Arial, sans-serif;">
  To address these challenges, this work introduces a novel hyperspectral blood detection dataset and investigates eight dimensionality reduction techniques as preprocessing methods. These methods are evaluated using state-of-the-art fast and compact 3D Convolutional Neural Network (CNN) and Hybrid CNN models. Experimental results underscore the complexities involved in hyperspectral blood detection and suggest avenues for future research. Additionally, the paper emphasizes the value of <strong style="color: #4CAF50;">Factor Analysis</strong> as a statistical approach to uncover underlying factors explaining the patterns and relationships among observed variables.
</p>

<h2> Methodology </h2>
  
<div align="center"> 
  <img width="200" src="https://raw.githubusercontent.com/sreenijakurra/Hyperspectral-Blood-Stain-Image-Classification/main/HSI img1.png" />
</div>




<h2>Key Contributions</h2>
<ul>
    <li>Explored multiple dimensionality reduction methods (feature extraction and selection) to improve processing efficiency for CNN models.</li>
    <li>Demonstrated improved performance using advanced dimensionality reduction methods.</li>
    <li>Established that <strong>Factor Analysis (FA)</strong> outperforms traditional techniques (PCA, LDA, iPCA) in classifying bloodstains.</li>
</ul>

<h2>Data Collection</h2>
<p>
Our dataset includes hyperspectral images of bloodstains and visually similar substances. To realistically simulate forensic challenges, we created Scene-E, a mock crime scene incorporating blood, acrylic paint, ketchup, and other substances on varied backgrounds (metal, wood, plastic, fabrics). This diverse setup tests the robustness of blood detection algorithms.
</p>

<h2>Dimensionality Reduction</h2>
<p>
Dimensionality Reduction (DR) techniques decrease the complexity of hyperspectral data by selecting essential spectral bands, reducing computational load, and improving model accuracy. Initially, the dataset dimensions (696x520x128) were reduced to 113 spectral bands and further compressed to 15 bands using several methods:
</p>

<ul>
    <li><strong>Factor Analysis (FA):</strong> Groups correlated bands to fewer factors.</li>
    <li><strong>Principal Component Analysis (PCA):</strong> Maximizes variance by projecting data onto orthogonal axes.</li>
    <li><strong>Incremental PCA (iPCA):</strong> Efficient PCA suited for non-Gaussian data distributions.</li>
    <li><strong>Sparse PCA (SPCA):</strong> Enhances interpretability by encouraging sparse solutions.</li>
    <li><strong>Singular Value Decomposition (SVD):</strong> Retains significant singular values through matrix factorization.</li>
    <li><strong>Fast Independent Component Analysis (Fast ICA):</strong> Extracts statistically independent components through nonlinear separation.</li>
    <li><strong>Non-negative Matrix Factorization (NMF):</strong> Enhances interpretability by factorizing into non-negative matrices.</li>
    <li><strong>Gaussian Random Projection (GRP):</strong> Utilizes random projections for efficient dimensionality reduction.</li>
</ul>

</body>
</html>

<h2>Conclusion</h2>
<p>
According to the analysis, Factor Analysis dimensionality reduction method gives the highest accuracy out of all other DR techniques. The fast and compact 3D CNN model achieved an accuracy of 99.34% for the calculation of FA, while the hybrid CNN model achieved an even higher accuracy of 99.70%. In terms of time complexity of the dimension reduction, GRP method proves to be highly efficient, however gives tremendously less accuracy than the other methods. For further analysis, if we need to consider a DR method based on both accuracy and time complexity, PCA, SPCA, SVD prove to be better alternatives as they take less time to execute and give better results when compared to all other DR methods.
</p>
