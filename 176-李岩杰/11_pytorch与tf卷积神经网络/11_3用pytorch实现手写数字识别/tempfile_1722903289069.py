   with torch.no_grad():
       correct = 0
       total = 0
       for images, labels in test_dataloader:  # 假设 test_dataloader 是测试数据加载器
           outputs = model(images)
           _, predicted = torch.max(outputs.data, 1)
           total += labels.size(0)
           correct += (predicted == labels).sum().item()

       print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
   