#To get kth bit
def get_bit(n,k):
    if(k<=0):
        return 0
    if((n&(1<<k-1))!=0):
        return 1
    return 0

#To count number of 1's in binary representation
#Brian-Kernningham Algorithm
def countSetBits(n):
    cnt=0
    while(n>0):
        n = n&(n-1)
        cnt+=1
    return cnt

#Check Power of 2
def powerOf2(n):
    if(countSetBits(n)==1):
        return True
    return False

#Checks if there is any element occuring odd number of times
def oddOccurence(arr, n):
    xor=arr[0]
    for i in range(1,n):
        xor = xor^arr[i]
    if(xor==0):
        return False
    return True

#If there are two numbers in an array that occur odd number of times, this function will return both
def twoOddOccurences(arr, n):
    xor=arr[0]
    x=0
    y=0

    for i in range(1,n):
        xor = xor^arr[i]
    #This xor is nothing but x^y
    
    #set_bit is the rightmost set bit in xor
    #A set bit in xor means that bit is different for x and y
    set_bit= xor & ~(xor-1)
    for i in range(n):
        if(arr[i] & set_bit):
            x = x^arr[i]
        else:
            y = y^arr[i]

    print("Odd occuring numbers:",x,y)
    
#Driver code
N,K = list(map(int, input("Enter n(number to perform all o/p) and k(bit to check): ").split()))
print()
print("The value of given bit in number:",get_bit(N,K))
print("No of 1's in binary represantation:",countSetBits(N))
print("Is the number power of 2?:",powerOf2(N))

print()
arr = list(map(int, input("Enter array elements: ").split()))
n=len(arr)
print("Does array have any element odd no of times?:",oddOccurence(arr,n))
print()

arr2 = list(map(int, input("Enter array elements where two elements are odd no of times: ").split()))
n2=len(arr2)
twoOddOccurences(arr2,n2)
