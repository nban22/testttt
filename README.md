# INTRODUCTION TO MACHINE LEARNING

## Table of contents

1. [Machine Learning](#1-Machine-Learing)
2. [Learning Agent]()
3. [Data]()
4. [Exploration]()
5. [Type of Learning]()
6. [Successes of Machine Learning]()

---

# 1. Machine Learing

-   Machine Learing
-   Interpretable Machine Learning

## What is the Learning Problem?

    Contept 1

    Learning -> Improving with experience at some task
    - Improve over task T
    - with respect to performance measure P
    - based on experience E

## Inductive Learning

Inductive learning (học suy diễn) là một phương pháp học trong trí tuệ nhân tạo và học máy, nơi một mô hình được xây dựng bằng cách tổng quát hóa từ các quan sát hoặc dữ liệu cụ thể. Quá trình này thường diễn ra theo cách sau:

-   Quan sát dữ liệu cụ thể
-   Tìm kiếm mẫu
-   Áp dụng cho dữ liệu mới

Ví dụ, trong bài toán phân loại email là "spam" hay "không phải spam", một hệ thống học suy diễn có thể học từ các email đã được gán nhãn trước đó. Hệ thống sẽ tìm ra những đặc điểm chung của các email thuộc từng nhóm và áp dụng các quy tắc này để phân loại các email mới.

![alt text](image.png)

Inductive learning đối lập với deductive learning (học diễn dịch), trong đó kiến thức được suy ra từ những nguyên tắc hoặc quy tắc đã có sẵn thay vì từ các ví dụ cụ thể.

## Machine Learning vs. Traditional Programming

-   Traditional Programming

    > Cách tiếp cận: Trong lập trình truyền thống, con người viết mã dựa trên các quy tắc rõ ràng để máy tính thực hiện nhiệm vụ. Người lập trình cần cung cấp một tập hợp các quy tắc, điều kiện, và logic để xử lý các đầu vào và tạo ra đầu ra mong muốn

    -   Input + Rules (quy tắc) → Output

-   Machine Learning

    > Cách tiếp cận: Machine learning sử dụng dữ liệu và thuật toán để máy tính tự học các quy tắc hoặc mô hình mà không cần người lập trình cung cấp trực tiếp. Thay vì viết ra từng quy tắc, người lập trình cung cấp một lượng lớn dữ liệu, và hệ thống tự tìm ra các mẫu hoặc quy luật tiềm ẩn từ dữ liệu đó.

    -   Input + Output (Dữ liệu + Đầu ra mong muốn) → Learning Algorithm → Model (mô hình)

![alt text](image-1.png)

## Machine Learning vs. Agents

-   Machine Learning là một công cụ để học từ dữ liệu, còn Agents là các hệ thống thực hiện hành động và có thể học hỏi từ môi trường.

-   Machine Learning có thể được sử dụng trong các Agent (tác tử) để giúp chúng học từ kinh nghiệm và cải thiện các quyết định.

### So sánh Machine Learning và Agents:

| **Machine Learning**                                                  | **Agents (Tác tử)**                                           |
| --------------------------------------------------------------------- | ------------------------------------------------------------- |
| Học từ dữ liệu quá khứ để cải thiện mô hình dự đoán.                  | Nhận thức và tương tác trực tiếp với môi trường.              |
| Dữ liệu thường được cung cấp trước (offline learning).                | Tác tử phản ứng theo thời gian thực với môi trường.           |
| Tập trung vào các bài toán cụ thể như phân loại, hồi quy.             | Tập trung vào việc ra quyết định và hành động.                |
| Đa phần chỉ dựa trên dữ liệu đã có sẵn.                               | Học hỏi và ra quyết định dựa trên phản hồi từ môi trường.     |
| Có thể không cần trực tiếp tương tác với môi trường để học.           | Liên tục tương tác với môi trường để đạt mục tiêu.            |
| Thuật toán học phổ biến: học có giám sát, không giám sát, tăng cường. | Thuật toán phổ biến: học tăng cường (reinforcement learning). |


## Machine Learning vs. Artificial Intelligence

Phạm vi: AI là một lĩnh vực rộng lớn bao gồm nhiều nhánh khác nhau như:

    - Machine Learning (học máy),
    - Natural Language Processing (xử lý ngôn ngữ tự nhiên),
    - Computer Vision (thị giác máy tính),
    - Expert Systems (hệ thống chuyên gia),
    - Robotics (robot học)

- AI là một khái niệm rộng hơn, tập trung vào việc xây dựng các hệ thống có thể thực hiện những nhiệm vụ đòi hỏi trí thông minh.

- Machine Learning là một phương pháp bên trong AI, giúp các hệ thống học từ dữ liệu và tự động cải thiện khả năng của chúng.

### So sánh giữa AI và ML

| **Artificial Intelligence (AI)**                              | **Machine Learning (ML)**                               |
|---------------------------------------------------------------|---------------------------------------------------------|
| AI là lĩnh vực rộng lớn liên quan đến việc tạo ra các hệ thống thông minh có khả năng học hỏi và ra quyết định. | ML là một nhánh cụ thể của AI tập trung vào việc học từ dữ liệu để cải thiện khả năng dự đoán. |
| AI bao gồm nhiều nhánh như xử lý ngôn ngữ tự nhiên, thị giác máy tính, và học máy. | ML chỉ là một phần của AI, tập trung vào xây dựng các thuật toán học từ dữ liệu. |
| AI có thể bao gồm các hệ thống không dựa vào học máy (ví dụ: hệ thống chuyên gia). | ML yêu cầu dữ liệu để học và tự động điều chỉnh mô hình dựa trên dữ liệu đó. |
| AI hướng đến việc tạo ra các hệ thống thông minh có thể bắt chước hoặc vượt qua trí thông minh con người. | ML tập trung vào việc làm cho các hệ thống có khả năng tự cải thiện hiệu suất qua thời gian nhờ dữ liệu. |


## Interpretable Machine Learning - Explainable Artificial Intelligence

- là hai khái niệm liên quan đến việc làm cho các mô hình máy học và trí tuệ nhân tạo dễ hiểu hơn cho con người, giúp giải thích cách chúng đưa ra quyết định

    - IML tập trung vào việc làm cho các mô hình máy học trở nên dễ hiểu và minh bạch hơn.
    - XAI bao gồm IML và mở rộng hơn với các kỹ thuật giải thích cho các mô hình AI phức tạp, nhằm giúp các hệ thống AI hoạt động trong các ứng dụng thực tế mà vẫn có thể giải thích và minh bạch.

### So sánh tổng quát:

| **Interpretable Machine Learning (IML)** | **Explainable Artificial Intelligence (XAI)** |
| --- | --- |
| Tập trung vào việc làm cho các mô hình máy học dễ hiểu hơn. | Tập trung vào việc làm cho các hệ thống AI có thể giải thích được. |
| Phạm vi chủ yếu là máy học, đặc biệt là các mô hình toán học. | Phạm vi rộng hơn, bao gồm cả các hệ thống AI ngoài máy học. |
| Thường sử dụng các mô hình đơn giản hoặc các công cụ như LIME, SHAP để giải thích. | Bao gồm cả giải thích cho các mô hình phức tạp như học sâu, học tăng cường. |
| Tập trung vào tính minh bạch và dễ hiểu của mô hình. | Tập trung vào sự minh bạch và khả năng giải thích hành vi AI. |


## The Explainable Artificial Intelligence (cont.)

![alt text](image-2.png)

## Human vs. Mode
The challenge is people don’t understand the system and the system doesn’t
understand the people.  
- We need models for the system to use to understand and explain things to
the human.
- We also need models in the human’s head about what the system does

![alt text](image-3.png)

# 2. Learning Agent 

## Agents Need Learning
- Learning is essential for unknown environments, i.e., when designer lacks omniscience  
( Học tập là điều cần thiết trong môi trường chưa biết, nghĩa là khi người thiết kế không toàn tri)
    
    Ví dụ: Một robot được lập trình để hoạt động trong môi trường nhà có thể không được thiết kế để xử lý mọi thay đổi, như sự xuất hiện của vật cản mới hoặc thay đổi bố trí. Nếu robot có khả năng học hỏi từ kinh nghiệm và thay đổi, nó sẽ điều chỉnh cách hoạt động sao cho phù hợp hơn với thực tế.

- Learning is useful as a system construction method, i.e., expose the agent to reality rather than trying to write it down  
(Học tập là một phương pháp hữu ích để xây dựng hệ thống, tức là đưa tác nhân vào thực tế thay vì cố gắng lập trình tất cả)

    Ví dụ: Trong một game, thay vì lập trình tất cả các tình huống mà một tác nhân AI có thể gặp, người phát triển có thể để tác nhân chơi game nhiều lần và học từ những lần thất bại để cải thiện cách chơi của mình.

- Learning modifies the agent’s decision mechanisms to improve performance  
(Học tập thay đổi các cơ chế quyết định của tác nhân để cải thiện hiệu suất)

    Ví dụ: Một chiếc xe tự lái sử dụng học máy có thể không lái xe hoàn hảo trong những lần đầu tiên. Tuy nhiên, qua thời gian và sau nhiều lần lái trong các điều kiện khác nhau, chiếc xe học cách điều chỉnh các quyết định, như cách giảm tốc độ khi có vật cản phía trước, hay cách điều chỉnh góc quay, để cải thiện hiệu suất lái và giảm thiểu rủi ro tai nạn.  


> Phần này giải thích rằng các agents cần học để hoạt động tốt hơn trong những môi trường không chắc chắn hoặc quá phức tạp để lập trình toàn diện. Thay vì viết ra tất cả các quy tắc và hành vi, việc để các agents tự học từ thực tế sẽ giúp chúng điều chỉnh và cải thiện cách ra quyết định, từ đó nâng cao hiệu suất tổng thể. Learning giúp các agents trở nên linh hoạt và thích nghi với môi trường thực tế một cách thông minh hơn.


## Forms of Learning
Any component of an agent can be improved by learning from data. The
improvements, and the techniques used to make them, depend on four major
factors:
- Which *component* is to be improved.
- What *prior knowledge* the agent already has.
- What *representation* is used for the data and the component.
- What *feedback* is available to learn from.

## Components to be learned

The components of these agents include:

1. direct mapping from conditions on the current state to actions.
2. A means to infer relevant properties of the world from the percept sequence.
3. Information about the way the world evolves and about the results of possible
actions the agent can take.
4. Utility information indicating the desirability of world states.
5. Action-value information indicating the desirability of actions.
6. Goals that describe classes of states whose achievement maximizes the
agent’s utility.

Each of these components can be learned.

## Representation and prior knowledge
- Representation là cách mà dữ liệu được biểu diễn trong mô hình AI hoặc máy học, và việc lựa chọn cách biểu diễn đúng rất quan trọng để giúp mô hình học tốt.
- Prior knowledge là kiến thức sẵn có trước đó, giúp mô hình học tốt hơn và nhanh hơn, đặc biệt trong những trường hợp thiếu dữ liệu hoặc vấn đề phức tạp.

Representations can be
- Functions with inputs, a vector of attribute values, and outputs, either a
continuous numerical value or a discrete value
- Functions and prior knowledge composed of first-order logic sentences
- Bayesian networks

## Feedback to learn from
There are three types of feedback that determine the three main types of learning:

- In **unsupervised learning** the agent learns patterns in the input even though
    - Trong học không giám sát, tác nhân nhận dữ liệu đầu vào mà không có phản hồi hoặc nhãn cụ thể. Mục tiêu chính là khám phá các mẫu, cấu trúc hoặc nhóm trong dữ liệu.
    - Ứng dụng: 
        - Phân cụm: Nhóm các mục tương tự (ví dụ: phân khúc khách hàng trong marketing).
        - Giảm chiều: Kỹ thuật như PCA (Phân tích thành phần chính) để đơn giản hóa tập dữ liệu trong khi vẫn bảo tồn thông tin quan trọng.
        - Phát hiện bất thường: Xác định các điểm dữ liệu khác thường khác biệt đáng kể so với phần lớn (ví dụ: phát hiện gian lận).


no explicit feedback is supplied.
- In **reinforcement learning** the agent learns from a series of
reinforcements—rewards or punishments.
    - Trong học tăng cường, tác nhân học cách đưa ra quyết định bằng cách tương tác với môi trường. Tác nhân nhận phản hồi dưới dạng phần thưởng (tăng cường tích cực) hoặc hình phạt (tăng cường tiêu cực) dựa trên hành động của nó.
    - Ứng dụng:
        - Chơi game: Các thuật toán như AlphaGo học để chơi game bằng cách cạnh tranh với chính nó.
        - Robot: Đào tạo robot để thực hiện nhiệm vụ thông qua thử nghiệm và sai sót.
        - Xe tự lái: Học chiến lược lái xe tối ưu thông qua mô phỏng và trải nghiệm thực tế.

- In **supervised learning** the agent observes some example input–output pairs and learns a function that maps from input to output
    - Trong học có giám sát, tác nhân được đào tạo bằng cách sử dụng dữ liệu có nhãn, trong đó mỗi đầu vào được liên kết với một đầu ra đã biết. Mục tiêu là học một hàm ánh xạ từ đầu vào đến đầu ra
    - Ứng dụng:
        - Phân loại: Gán nhãn cho dữ liệu đầu vào (ví dụ: phát hiện thư rác trong email).
        - Hồi quy: Dự đoán các giá trị liên tục dựa trên các đặc trưng đầu vào (ví dụ: dự đoán giá nhà).
        - Nhận diện hình ảnh: Xác định các đối tượng trong hình ảnh dựa trên dữ liệu huấn luyện có nhãn.

    > - In semi-supervised learning we are given a few labeled examples and must make what we can of a large collection of unlabeled examples.
    >   - Học bán giám sát là một phương pháp kết hợp giữa học có giám sát và không giám sát. Nó tận dụng một lượng nhỏ dữ liệu có nhãn cùng với một tập hợp lớn các dữ liệu không có nhãn.
    >   - Ứng dụng:
    >       - Phân loại văn bản: Sử dụng một vài tài liệu có nhãn để phân loại một tập hợp lớn văn bản.
    >       - Phân loại hình ảnh: Đào tạo trên một tập hợp hình ảnh có nhãn nhỏ và một số lượng lớn hình ảnh không có nhãn để cải thiện hiệu suất mô hình.
    >       - Chẩn đoán y tế: Sử dụng các hồ sơ y tế có nhãn hạn chế để phân loại một tập dữ liệu bệnh nhân lớn hơn.


## Machine Learning Workflow

![alt text](image-4.png)

1. Collect available data.
2. Clean and transform that data. If you’re collecting data that is missing
values, then you need to clean and transform that data until it’s in the form
machine learning requires.
3. Explore and visualize the data to make sure it is encoding what you expect it
to encode. Build a model on training data.
4. Evaluate the model test data.
5. Deploy the model on un-seen data.
6. Monitor the model

# 3. Data

## Types of Data

Record
- Relational records
- Data matrix: numerical matrix
- Document data: text documents
- Transaction data

Graph and network
- World Wide Web
- Social or information networks
- Molecular Structures

Ordered  
- Video data: sequence of images
- Temporal data: time-series
- Sequential Data: transaction sequences
- Genetic sequence data

Spatial, image and multimedia
- Spatial data: maps
- Image data
- Video data

## Data Sets

Concept 2 

    - Data sets are made up of data objects.
    - A data object represents an entity (also called samples, examples, instances, data points, objects, tuples).
    - Data objects are described by attributes.

## Attributes
Concept 3

    An attribute (also called dimension, feature, variable) is a data field, representing a characteristic or feature of a data object.

| Data set | Attributes | 
|----------|------------| 
Sales database | customers, store items, sales
Medical database | patients, treatments
University database | students, professors, courses

## Attribute Types

Nominal: categories, states, or “names of things”
- hair color = {auburn, black, blond, brown, grey, red, white}
- marital status, occupation, ID numbers, zip codes

Binary: nominal attribute with only two states
- gender (symmetric: both outcomes equally important)
- medical test (positive vs. negative) (asymmetric: outcomes not equally
important)

Ordinal: values have a meaningful order but magnitude between successive
values is not known.
- size = {small, medium, large}
- grades, rankings

> Các loại thuộc tính này là rất quan trọng trong việc xây dựng mô hình học máy, vì chúng ảnh hưởng đến cách mà dữ liệu được xử lý và phân tích. Việc hiểu rõ các thuộc tính sẽ giúp người nghiên cứu chọn lựa phương pháp thích hợp để xử lý và phân loại dữ liệu trong quá trình phát triển mô hình.

## Numeric Attribute Types

- Quantity (integer or real-valued)
- Interval scale: Measured on a scale of equal-sized units. No true zero-point.
Values have order
    - temperature in Celsius or Fahrenheit
    - calendar dates
- Ratio scale: Inherent zero-point
    - temperature in Kelvin
    - length
    - counts
    - monetary quantitie

## Discrete vs. Continuous Attributes
(Các loại thuộc tính Rời rạc (Discrete) và Liên tục (Continuous))

- Discrete Attribute: Has only a finite or countably infinite set of values
    - zip codes
    - profession  

    Note: Sometimes, represented as integer variables; binary attributes are a
special case of discrete attributes
- Continuous Attribute: Has real numbers as attribute values
    - temperature
    - height, or weight

> Việc phân loại thuộc tính thành rời rạc và liên tục là rất quan trọng trong việc phát triển mô hình học máy, vì nó ảnh hưởng đến cách mà dữ liệu được xử lý và các thuật toán nào sẽ được sử dụng. Các thuộc tính rời rạc thường liên quan đến các bài toán phân loại, trong khi các thuộc tính liên tục thường liên quan đến các bài toán hồi quy, nơi mà sự thay đổi liên tục giữa các giá trị là điều quan trọng.

## Sample datasets

![alt text](image-5.png)

![alt text](image-6.png)

## Large Language Models

![alt text](image-7.png)

## Tensor-based Attributes

Concept 4  

    A tensor is a generalized matrix, a finite table of numerical values indexed along several discrete dimensions.
        - A 0d tensor is a scalar
        - A 1d tensor is a vector (e.g. a sound sample)
        - A 2d tensor is a matrix (e.g. a grayscale image)
        - A 3d tensor (e.g. a multi-channel image)
        - A 4d tensor (e.g. a sequence of multi-channel images)

![alt text](image-8.png)

# 4. Exploration

## Data Visualization

Concept 5
    
    Visualization is the conversion of data into a visual or tabular format so that the characteristics of the data and the relationships among data items or attributes can be analyzed or reported.

Humans have a well developed ability to analyze large amounts of information
that is presented visually
- Can detect general patterns and trends
- Can detect outliers and unusual patterns

![alt text](image-9.png)

# 5. Type of Learnings

## Basic Premise of Learning

`Using a set of observations to uncover an underlying process or rule` 

>                   broad premise => many variations

- Supervised learning
- Unsupervised learning
- Reinforcement learning

## Supervised Learning

![alt text](image-10.png)

## Unsupervised Learning

![alt text](image-11.png)

## Reinforcement Learning

![alt text](image-12.png)

# 6. Successes of Machine Learning

- Vision
- Language
- Games
- Robotics
- Other

## 6.1 Vision

### Detection and Segmentation

![alt text](image-13.png)

### Face recognition

![alt text](image-14.png)

### Image generation

![alt text](image-15.png)

### DeepFakes

![alt text](image-16.png)

## 6.1 Translation

![alt text](image-17.png)

### Text Generation

![alt text](image-18.png)

### ChatBot

![alt text](image-19.png)

### Speech Applications

![alt text](image-20.png)

### Sketch2Code

![alt text](image-21.png)

### Auto-captioning

![alt text](image-22.png)

### Text-to-Image

![alt text](image-23.png)

![alt text](image-24.png)

## 6.3 Games

![alt text](image-25.png)

## 6.4 Robotics

### Self-driving cars

![alt text](image-26.png)

## 6.5 Others

### Solving scientific problems

![alt text](image-27.png)

# References

- Goodfellow, I., Bengio, Y., and Courville, A. (2016).

    Deep learning.  
    MIT press.
- Lê, B. and Tô, V. (2014).

    Cở sở trí tuệ nhân tạo.   
    Nhà xuất bản Khoa học và Kỹ thuật.  
- Russell, S. and Norvig, P. (2021).

    Artificial intelligence: a modern approach.  
    Pearson Education Limited.
