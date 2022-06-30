# Week 1: Supervised Machine Learning: Classification and Regression

## Supervised vs. Unsupervised Machine Learning

### What is Machine Learning?

 > "Field of study that gives computers the ability to learn without being explicitly programmed" - Arthur Samuel (1959)

- In general the more opportunities you give a learning algorithm to learn, the better it will perform 
- Types of machine learning algorithms:
  - Supervised Learning 
  - Unsupervised Learning
  - Recommender Systems 
  - Reinforcement Learning

### Supervised Learning: part 1

- Supervised learning: 
  
  ```mermaid
  flowchart LR
  subgraph input;
  X;
  end;
  subgraph output;
  Y;
  end;
  input --> output;
  ```

- You can train the model by giving it, correct output labels (Y) for every input (X), such that for an unknown input X_1 the algorithm can give a fairly accurate output label Y_1

