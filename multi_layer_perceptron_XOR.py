import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
digits = load_digits() 

print('image[0]: \n{}'.format(digits.images[0]))
print('target[0]: \n{}'.format(digits.target[0]))
print('전체 샘플의 수: {}'.format(len(digits.images)))

images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:5]):  # 5개의 샘플만 출력
    plt.subplot(2, 5, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('sample: %i' %label)
plt.show()

for i in range(5):
    print