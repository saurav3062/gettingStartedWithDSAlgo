#include<iostream>
#include<vector>
#include<map>
#include<set>
#include<queue>
#include<stack>
#include<algorithm>
#include<functional>
#include<math.h>
#include<string>
#include <climits>
#include<unordered_set>
#include<unordered_map>
#include <ranges>
using namespace std;
struct ListNode {
	int val;
	ListNode* next;
	ListNode* random;
	ListNode* child;
	ListNode() : val(0), next(nullptr) {}
	ListNode(int x) : val(x), next(nullptr) {}
	ListNode(int x, ListNode* next) : val(x), next(next) {}
	
};

void powerSet(string s, int i, string curr) {
	if (i == s.length()) {
		cout << curr << endl;
		return;
	}
	powerSet(s, i + 1, curr + s[i]);
	powerSet(s, i + 1, curr);
}
int josephus(int n, int k) {
	if (n == 1)
		return 0;
	return (josephus(n - 1, k) + k) % n;
}
int matrixPath(int n, int m) {
	if (n == 1 || m == 1)
		return 1;
	return matrixPath(n - 1, m) + matrixPath(n, m - 1);
}
void permute(string s, int l, int r) {
	if (l == r) {
		cout << s << endl;
		return;
	}
	for (int i = l; i <= r; i++) {
		swap(s[l], s[i]);
		permute(s, l + 1, r);
		swap(s[l], s[i]);
	}
}

bool isSafe(vector<vector<int>>& board, int row, int col) {
	for (int i = 0; i < row; i++) {
		if (board[i][col])
			return false;
	}
	for (int i = row, j = col; i >= 0 && j >= 0; i--, j--) {
		if (board[i][j])
			return false;
	}
	for (int i = row, j = col; i >= 0 && j < board.size(); i--, j++) {
		if (board[i][j])
			return false;
	}
	return true;
}
bool nQueen(vector<vector<int>>& board, int row) {
	if (row == board.size())
		return true;
	for (int col = 0; col < board.size(); col++) {
		if (isSafe(board, row, col)) {
			board[row][col] = 1;
			if (nQueen(board, row + 1))
				return true;
			board[row][col] = 0;
		}
	}
	return false;
}
bool solveSudoku(vector<vector<int>>& board, int row, int col) {
	if (row == board.size() - 1 && col == board.size())
		return true;
	if (col == board.size()) {
		row++;
		col = 0;
	}
	if (board[row][col] != 0)
		return solveSudoku(board, row, col + 1);
	for (int i = 1; i <= board.size(); i++) {
		if (isSafe(board, row, col)) {
			board[row][col] = i;
			if (solveSudoku(board, row, col + 1))
				return true;
		}
		board[row][col] = 0;
	}
	return false;
} //  this code wont work refer geeks for geeks
void printBoard(vector<vector<int>>& board) {
	for (int i = 0; i < board.size(); i++) {
		for (int j = 0; j < board.size(); j++)
			cout << board[i][j] << " ";
		cout << endl;
	}
}

int mooresVoting(vector<int>& nums) {
	int count = 0, candidate = 0;
	for (int i = 0; i < nums.size(); i++) {
		if (count == 0)
			candidate = nums[i];
		if (candidate == nums[i])
			count++;
		else
			count--;
	}
	return candidate;
}
int maxSumSubArray(vector<int>& nums) {
	int maxSum = INT_MIN, currSum = 0;
	for (int i = 0; i < nums.size(); i++) {
		currSum += nums[i];
		if (currSum > maxSum)
			maxSum = currSum;
		if (currSum < 0)
			currSum = 0;
	}
	return maxSum;
}// kadens algorithm

int stockBuySell(vector<int>& nums) {
	vector<int> aux(nums.size());
	aux[nums.size()-1] = nums[nums.size()-1];
	for (int i = nums.size() - 2; i >= 0; i--) {
		aux[i] = max(aux[i + 1], nums[i]);
	}
	int maxProfit = INT_MIN;
	for (int i = 0; i < nums.size(); i++) {
		maxProfit = max(maxProfit, aux[i] - nums[i]);
	}
	return maxProfit;
}
int maxProfit(vector<int>& prices) {
	int minPrice = INT_MAX, maxProfit = 0;
	for (int i = 0; i < prices.size(); i++) {
		if (prices[i] < minPrice)
			minPrice = prices[i];
		else if (prices[i] - minPrice > maxProfit)
			maxProfit = prices[i] - minPrice;
	}
	return maxProfit;
}//optimized solution for stock buy sell in which we can buy and sell only once.
int maxProfit2(vector<int>& prices) {
	int maxProfit = 0;
	for (int i = 1; i < prices.size(); i++) {
		if (prices[i] > prices[i - 1])
			maxProfit += prices[i] - prices[i - 1];
	}
	return maxProfit;
}// optimized solution for stock buy sell 2 in which we can buy and sell multiple times in a day and we have to maximize the profit in the end of the day.
int rainWaterTrapping(vector<int>& nums) {
	int left = 0, right = nums.size() - 1, leftMax = 0, rightMax = 0, water = 0;
	while (left < right) {
		if (nums[left] < nums[right]) {
			if (nums[left] > leftMax)
				leftMax = nums[left];
			else
				water += leftMax - nums[left];
			left++;
		}
		else {
			if (nums[right] > rightMax)
				rightMax = nums[right];
			else
				water += rightMax - nums[right];
			right--;
		}
	}
	return water;
} // optimized solution for rain water trapping problem.
int rainWaterTrapping2(vector<int>& nums) {
	int n = nums.size();
	vector<int> left(n), right(n);
	left[0] = nums[0];
	right[n - 1] = nums[n - 1];
	for (int i = 1; i < n; i++) {
		left[i] = max(left[i - 1], nums[i]);
		right[n - i - 1] = max(right[n - i], nums[n - i - 1]);
	}
	int water = 0;
	for (int i = 0; i < n; i++) {
		water += min(left[i], right[i]) - nums[i];
	}
	return water;
} // brute force solution for rain water trapping problem.
void bubbleSort(vector<int>& nums) {
	for (int i = 0; i < nums.size(); i++) {
		bool swapped = false;
		for (int j = 0; j < nums.size() - i - 1; j++) {
			if (nums[j + 1] < nums[j]) {
				swapped = true;
				swap(nums[j], nums[j + 1]);
			}
		}
		if (!swapped)
			break;
	}
}
void insertionSort(vector<int>& nums) {
	for (int i = 1; i < nums.size(); i++) {
		int temp = nums[i];
		int j = i-1;
		while (j >= 0 && nums[j]>temp) {
			nums[j+1]= nums[j];
			j--;
		}
		nums[j+1] = temp;
	}
}
void selectionSort(vector<int>& nums) {
	for (int i = 0; i < nums.size(); i++) {
		int minIndex = i;
		for (int j = i + 1; j < nums.size(); j++) {
			if (nums[j] < nums[minIndex])
				minIndex = j;
		}
		swap(nums[i], nums[minIndex]);
	}
}

int partition(vector<int>& nums, int low, int high) {
	int pivot = nums[high];
	int i = low - 1;
	for (int j = low; j < high; j++) {
		if (nums[j] < pivot) {
			i++;
			swap(nums[i], nums[j]);
		}
	}
	swap(nums[i + 1], nums[high]);
	return i + 1;
}
void quickSort(vector<int>& nums, int low, int high) {
	if (low < high) {
		int pivot = partition(nums, low, high);
		quickSort(nums, low, pivot - 1);
		quickSort(nums, pivot + 1, high);
	}
}

void merge(vector<int>& nums, int low, int mid, int high) {
	int n1 = mid - low + 1;
	int n2 = high - mid;
	vector<int> left(n1), right(n2);
	for (int i = 0; i < n1; i++)
		left[i] = nums[low + i];
	for (int i = 0; i < n2; i++)
		right[i] = nums[mid + 1 + i];
	int i = 0, j = 0, k = low;
	while (i < n1 && j < n2) {
		if (left[i] <= right[j])
			nums[k++] = left[i++];
		else
			nums[k++] = right[j++];
	}
	while (i < n1)
		nums[k++] = left[i++];
	while (j < n2)
		nums[k++] = right[j++];
}
void mergeSort(vector<int>& nums, int low, int high) {
	if (low < high) {
		int mid = low + (high - low) / 2;
		mergeSort(nums, low, mid);
		mergeSort(nums, mid + 1, high);
		merge(nums, low, mid, high);
	}
}
int binarySearch(vector<int>& nums, int low, int high, int target) {
	if (low <= high) {
		int mid = low + (high - low) / 2;
		if (nums[mid] == target)
			return mid;
		else if (nums[mid] > target)
			return binarySearch(nums, low, mid - 1, target);
		else
			return binarySearch(nums, mid + 1, high, target);
	}
	return -1;
}
int binarySearchIterative(vector<int>& nums, int target) {
	int low = 0, high = nums.size() - 1;
	while (low <= high) {
		int mid = low + (high - low) / 2;
		if (nums[mid] == target)
			return mid;
		else if (nums[mid] > target)
			high = mid - 1;
		else
			low = mid + 1;
	}
	return -1;
}
int searchInfinite(vector<int>& num, int key) {
	int low = 0;
	int high = 1;
	while (num[high]<key)
	{
		low = high;
		high = 2 * high;
	}
	return binarySearch(num, low, high, key);
}
int bSearch(vector<int>& num, int key) {
	int low = 0;
	int high = num.size() - 1;
	while (low <= high) {
		int mid = low + (high - low) / 2;
		if (num[mid] == key)
			return mid;
		else if (num[mid] >= num[low]) {
			if (key >= num[low] && key < num[mid])
				high = mid - 1;
			else
				low = mid + 1;
		}
		else {
			if (key > num[mid] && key <= num[high])
				low = mid + 1;
			else
				high = mid - 1;
		}
	}
	return -1;
} // the array is sorted and rotated.
void disElemWin(vector<int>& num, int k) {
	unordered_map<int, int> fre;
	for (int i = 0; i < k; i++)
		fre[num[i]]++;
	cout << fre.size() << " ";
	for (int i = k; i < num.size(); i++) {
		fre[num[i - k]]--;
		if (fre[num[i - k]] == 0)
			fre.erase(num[i - k]);
		fre[num[i]]++;
		cout << fre.size() << " ";
	}
}// distinct elements in every window of size k

void insertHeap(vector<int>& ele, int val) {
	int n = ele.size();
	ele.push_back(val);
	int i = n;
	while (i > 0 && ele[(i - 1) / 2] < ele[i]) {
		swap(ele[(i - 1) / 2], ele[i]);
		i = (i - 1) / 2;
	}
} // insert in max heap
void deleteHeap(vector<int>& ele) {
	ele[1] = ele[ele.size() - 1];
	ele.pop_back();
	int i = 1;
	int n = ele.size();
	while (i < n) {
		int left = ele[2 * i];
		int right = ele[2 * i + 1];
		if (ele[i] > left && ele[i] > right)
			break;
		else if (left > right) {
			swap(ele[i], ele[2 * i]);
			i = 2 * i;
		}
		else {
			swap(ele[i], ele[2 * i + 1]);
			i = 2 * i + 1;
		}
	}
} // delete head in max heap
void heapify(vector<int>& ele, int n, int i) {
	int largest = i;
	int l = 2 * i + 1;
	int r = 2 * i + 2;
	if (l<n && ele[l]>ele[largest])
		largest = l;
	if (r<n && ele[r]>ele[largest])
		largest = r;
	if (largest != i) {
		swap(ele[i], ele[largest]);
		heapify(ele, n, largest);
	}
}
void heapSort(vector<int>& ele) {
	int n = ele.size();
	for (int i = n / 2 - 1; i >= 0; i--)
		heapify(ele, n, i);
	for (int i = n - 1; i >= 0; i--) {
		swap(ele[0], ele[i]);
		heapify(ele, i, 0);
	}
} // heap sort
int minCostToConnectRopes(vector<int>& ropes) {
priority_queue<int, vector<int>, greater<int>> pq(ropes.begin(), ropes.end());
	int cost = 0;
	while (pq.size() > 1) {
		int first = pq.top();
		pq.pop();
		int second = pq.top();
		pq.pop();
		cost += first + second;
		pq.push(first + second);
	}
	return cost;
} // min cost to connect ropes

ListNode* reverseLinkedList(ListNode* head) {
	ListNode* curr = head;
	ListNode* prev = NULL;
	while (curr) {
		ListNode* next = curr->next;
		curr->next = prev;
		prev = curr;
		curr = next;
	}
	return prev;
}
ListNode* reverseLinkedListRecursion(ListNode* head) {
	if (!head || !head->next)
		return head;
	ListNode* rest = reverseLinkedListRecursion(head->next);
	head->next->next = head;
	head->next = NULL;
	return rest;
}
ListNode* detectCycle(ListNode* head) {
	ListNode* slow = head;
	ListNode* fast = head;
	while (fast && fast->next) {
		slow = slow->next;
		fast = fast->next->next;
		if (slow == fast)
			break;
	}
	return slow;
}
ListNode* detectFirstNode(ListNode* head) {
	ListNode* meet = detectCycle(head);
	ListNode* start = head;
	while (meet != start) {
		meet = meet->next;
		start = start->next;
	}
	return meet;
}
ListNode* duplicateWithRandomPointer(ListNode* head) {
	ListNode* curr = head;
	while (curr) {
		ListNode* next = curr->next;
		ListNode* dup = new ListNode(curr->val);
		curr->next = dup;
		dup->next = next;
		curr = next;
	}
	curr = head;
	while (curr) {
		if (curr->next != NULL) {
			curr->next->random = (curr->random != NULL) ? curr->random->next : NULL;
		}
		curr = curr->next->next;
	}
	ListNode* original = head;
	ListNode* copy = head->next;
	ListNode* temp = copy;
	while (original && copy) {
		original->next = original->next ? original->next->next : original->next;
		copy->next = copy->next ? copy->next->next : copy->next;
		original = original->next;
		copy = copy->next;
	}
	return temp;
}

vector<int> nextGreaterElement(vector<int>& nums) {
	vector<int> res(nums.size());
	stack<int> st;
	for (int i = nums.size() - 1; i >= 0; i--) {
		while (!st.empty() && st.top() <= nums[i])
			st.pop();
		res[i] = st.empty() ? -1 : st.top();
		st.push(nums[i]);
	}
	return res;
}
vector<int> nextSmallerElement(vector<int>& nums) {
	vector<int> res(nums.size());
	stack<int> st;
	for (int i = nums.size() - 1; i >= 0; i--) {
		while (!st.empty() && st.top() >= nums[i])
			st.pop();
		res[i] = st.empty() ? -1 : st.top();
		st.push(nums[i]);
	}
	return res;
}
vector<int> prevGreaterElement(vector<int>& nums) {
	vector<int> res(nums.size());
	stack<int> st;
	for (int i = 0; i < nums.size(); i++) {
		while (!st.empty() && st.top() <= nums[i])
			st.pop();
		res[i] = st.empty() ? -1 : st.top();
		st.push(nums[i]);
	}
	return res;
}

bool parenthesisMatching(string s) {
	stack<char> st;
	for (char c : s) {
		if (c == '(' || c == '{' || c == '[')
			st.push(c);
		else {
			if (st.empty())
				return false;
			if (c == ')' && st.top() != '(')
				return false;
			if (c == '}' && st.top() != '{')
				return false;
			if (c == ']' && st.top() != '[')
				return false;
			st.pop();
		}
	}
	return st.empty();
}
int largestRectangleArea(vector<int>& heights) {
	int n = heights.size();
	stack<int> st;
	int maxArea = 0;
	for (int i = 0; i <= n; i++) {
		while (!st.empty() && (i == n || heights[st.top()] > heights[i])) {
			int h = heights[st.top()];
			st.pop();
			int w = st.empty() ? i : i - st.top() - 1;
			maxArea = max(maxArea, h * w);
		}
		st.push(i);
	}
	return maxArea;
}

int maxAreaa(vector<int> heights) {
	int maxAns = 0;
	int n = heights.size();
	vector<int> prevSmaller(n, -1);
	vector<int> nextSmaller(n, n);
	for (int i = 0; i < n; i++) {
		int curr = (nextSmaller[i] - prevSmaller[i] - 1) * heights[i];
		maxAns = max(maxAns, curr);
	}
	return maxAns;
}
vector<int> prevSmaller(vector<int>& nums) {
	vector<int> res(nums.size());
	stack<int> st;
	for (int i = 0; i < nums.size(); i++) {
		while (!st.empty() && st.top() >= nums[i])
			st.pop();
		res[i] = st.empty() ? -1 : st.top();
		st.push(nums[i]);
	}
	return res;
}
vector<int> nextSmaller(vector<int>& nums) {
	vector<int> res(nums.size());
	stack<int> st;
	for (int i = nums.size() - 1; i >= 0; i--) {
		while (!st.empty() && st.top() >= nums[i])
			st.pop();
		res[i] = st.empty() ? -1 : st.top();
		st.push(nums[i]);
	}
	return res;
}
int largestAreaSubmatrix(vector<vector<int>> matrix) {
	vector<int> currRow(matrix[0]);
	int maxArea = maxAreaa(currRow);
	for (int i = 1; i < matrix.size(); i++) {
		for (int j = 0; j < matrix[0].size(); j++) {
			currRow[j] = matrix[i][j] == 0 ? 0 : currRow[j] + 1;
		}
		maxArea = max(maxArea, maxAreaa(currRow));
	}
	return maxArea;
}

int precedence(char c) {
	if (c == '+' || c == '-')
		return 1;
	if (c == '*' || c == '/')
		return 2;
	if (c=='^')
		return 3;
	return 0;
}
string infixToPostfix(string s) {
	string res = "";
	stack<char> st;
	for (char c : s) {
		if (c == '(')
			st.push(c);
		else if (c == ')') {
			while (!st.empty() && st.top() != '(') {
				res += st.top();
				st.pop();
			}
			st.pop();
		}
		else if (c == '+' || c == '-' || c == '*' || c == '/') {
			while (!st.empty() && st.top() != '(' && precedence(st.top()) >= precedence(c)) {
				res += st.top();
				st.pop();
			}
			st.push(c);
		}
		else
			res += c;
	}
	while (!st.empty()) {
		res += st.top();
		st.pop();
	}
	return res;
}

ListNode* flatten(ListNode* head) {
	queue<ListNode*> q;
	ListNode* curr = head;
	while (curr) {
		if (curr->next == NULL) {
			curr->next = q.front();
			q.pop();
		}
		if (curr->child != NULL) {
			q.push(curr->child);
		}
		curr = curr->next;
	}
	return head;
}
ListNode* flatten2(ListNode* head) {
	if (head == NULL) return NULL;
	ListNode* cur = head;
	ListNode* end = head;
	ListNode* temp;
	while (end->next != NULL) {
		end = end->next;
	}
	while (cur != end) {
		if (cur->child != NULL) {
			end->next = cur->child;
			temp = cur->child;
			while (temp->next != NULL) {
				temp = temp->next;
			}
			end = temp;
		}
		cur = cur->next;
	}
	return head;
}

vector<int> slidingWindowMaximum(vector<int>& nums, int k) {
	vector<int> res;
	deque<int> dq;
	for (int i = 0; i < nums.size(); i++) {
		if (!dq.empty() && dq.front() == i - k)
			dq.pop_front();
		while (!dq.empty() && nums[dq.back()] < nums[i])
			dq.pop_back();
		dq.push_back(i);
		if (i >= k - 1)
			res.push_back(nums[dq.front()]);
	}
	return res;
}

class Node {
	public:
	Node *left, *right; 
	int data, height;
	Node(int data) {
		this->data = data;
		left = right = NULL;
	}
	Node* createTree() {
		int data;
		cin >> data;
		if (data == -1)
			return NULL;
		Node* root = new Node(data);
		cout<<"Enter left child of "<<data<<endl;
		root->left = createTree();
		cout<<"Enter right child of "<<data<<endl;
		root->right = createTree();
		return root;
	}

	void inOrder(Node* root) {
		if (root == NULL) return;
		inOrder(root->left);
		cout << root->data << " ";
		inOrder(root->right);
	}
	void preOrder(Node* root) {
		if (root == NULL) return;
		cout << root->data << " ";
		preOrder(root->left);
		preOrder(root->right);
	}
	void postOrder(Node* root) {
		if (root == NULL) return;
		postOrder(root->left);
		postOrder(root->right);
		cout << root->data << " ";
	}
	void levelOrder(Node* root) {
		queue<Node*> q;
		q.push(root);
		while (!q.empty()) {
			Node* curr = q.front();
			q.pop();
			cout << curr->data << " ";
			if (curr->left != NULL)
				q.push(curr->left);
			if (curr->right != NULL)
				q.push(curr->right);
		}
	}
	void levelOrderLineByLine(Node* root) {
		queue<Node*> q;
		q.push(root);
		q.push(NULL);
		while (!q.empty()) {
			Node* curr = q.front();
			q.pop();
			if (curr == NULL) {
				cout << endl;
				if (!q.empty())
					q.push(NULL);
			}
			else {
				cout << curr->data << " ";
				if (curr->left != NULL)
					q.push(curr->left);
				if (curr->right != NULL)
					q.push(curr->right);
			}
		}
	}

};
Node* constructTree(vector<int> nums) {
	stack<Node*> st;
	for (int i = 0; i < nums.size(); i++) {
		if (nums[i] == -1) {
			Node* right = st.top();
			st.pop();
			Node* left = st.top();
			st.pop();
			Node* root = new Node(i);
			root->left = left;
			root->right = right;
			st.push(root);
		}
		else {
			Node* root = new Node(i);
			st.push(root);
		}
	}
	return st.top();
}
int heightOfTree(Node* root) {
	if (root == nullptr) return 0;
	return (1+max(heightOfTree(root->left), heightOfTree(root->right)));
}
int heightOfTree2(Node* root) {
	if (root == nullptr) return 0;
	int level = 0;
	std::deque<Node*> dq;
	dq.push_back(root);
	while (!dq.empty()) {
		int size = dq.size();
		for (int i = 0; i < size; i++) {
			Node* node = dq.front();
			dq.pop_front();
			if (node->left) dq.push_back(node->left);
			if (node->right) dq.push_back(node->right);
		}
		level++;
	}
	return level;	
}
int heightOfTree3(Node* root) {
	if (root == nullptr) return 0;

	int maxDepth = 0;
	std::stack<std::pair<Node*, int>> stk;
	stk.push({ root, 1 });

	while (!stk.empty()) {
		auto current = stk.top();
		stk.pop();

		Node* node = current.first;
		int depth = current.second;

		maxDepth = std::max(maxDepth, depth);

		if (node->right)
			stk.push({ node->right, depth + 1 });

		if (node->left)
			stk.push({ node->left, depth + 1 });
	}

	return maxDepth;
}
int heightOfTree4(Node* root) {
	if (root == nullptr) return 0;
	int level = 0;
	std::queue<Node*> q;
	q.push(root);
	while (!q.empty()) {
		int size = q.size();
		for (int i = 0; i < size; i++) {
			Node* node = q.front();
			q.pop();
			if (node->left) q.push(node->left);
			if (node->right) q.push(node->right);
		}
		level++;
	}
	return level;
}
int edgesOfTree(Node* root) {
	if (root == nullptr) return 0;
	return (heightOfTree(root) - 1);
}
int sizeOfTree(Node* root) {
	if (root == nullptr) return 0;
	return (1 + sizeOfTree(root->left) + sizeOfTree(root->right));
}
int sizeOfTree2(Node* root) {
	if (root == nullptr) return 0;
	int size = 0;
	std::stack<Node*> stk;
	stk.push(root);
	while (!stk.empty()) {
		Node* node = stk.top();
		stk.pop();
		size++;
		if (node->left) stk.push(node->left);
		if (node->right) stk.push(node->right);
	}
	return size;
}
int maxOfTree(Node* root) {
	if (root == NULL) {
		return INT_MIN;
	}
	return max(root->data, max(maxOfTree(root->left), maxOfTree(root->right)));
}
int maxOfTree2(Node* root) {
	if (root == NULL) {
		return INT_MIN;
	}
	int max = INT_MIN;
	std::stack<Node*> stk;
	stk.push(root);
	while (!stk.empty()) {
		Node* node = stk.top();
		stk.pop();
		if (node->data > max) max = node->data;
		if (node->left) stk.push(node->left);
		if (node->right) stk.push(node->right);
	}
	return max;
}
int minOfTree(Node* root) {
	if (root == NULL) {
		return INT_MAX;
	}
	return min(root->data, min(minOfTree(root->left), minOfTree(root->right)));
}
void printCurrentLevel(Node* root, int level) {
	if(root==nullptr) return;
	if(level==1) cout<<root->data<<" ";
	else if (level > 1) {
		printCurrentLevel(root->left, level-1);
		printCurrentLevel(root->right, level-1);
	}
}
void levelOrderTraversal(Node* root) {
	int h = heightOfTree(root);
	for (int i = 1; i <= h; i++) {
		printCurrentLevel(root, i);
	}// complexity O(n^2)
}
void levelOrderTraversal2(Node* root) {
	if (root == nullptr) return;
	std::queue<Node*> q;
	q.push(root);
	while (!q.empty()) {
		Node* node = q.front();
		q.pop();
		cout << node->data << " ";
		if (node->left) q.push(node->left);
		if (node->right) q.push(node->right);
	}// complexity O(n)
}

void topViewOfBinaryTree(Node* root) {
	if(root==nullptr) return;
	std::map<int, int> m;
	std::queue<pair<Node*, int>> q;
	q.push({root, 0});
	while(!q.empty()){
		auto p = q.front();
		q.pop();
		Node* node = p.first;
		int hd = p.second;
		if(m.find(hd)==m.end()) m[hd] = node->data;
		if(node->left) q.push({node->left, hd-1});
		if(node->right) q.push({node->right, hd+1});
	}
	for(auto x:m) cout<<x.second<<" ";
}// complexity O(nlogn)

void bottomViewOfBinaryTree(Node* root) {
	if(root==nullptr) return;
	std::map<int, int> m;
	std::queue<pair<Node*, int>> q;
	q.push({root, 0});
	while (!q.empty()) {
		auto p = q.front();
		q.pop();
		Node* node = p.first;
		int hd = p.second;
		m[hd] = node->data;
		if(node->left) q.push({node->left, hd-1});
		if(node->right) q.push({node->right, hd+1});
	}
	for(auto x:m) cout<<x.second<<" ";
}// complexity O(nlogn)

void convertToDLL(Node* root) {
	if(root==nullptr) return;
	static Node* prev = nullptr;
	static Node* head = nullptr; 
	convertToDLL(root->left);
	if(prev==nullptr) head = root;
	else {
		root->left = prev;
		prev->right = root;
	}
	prev = root;
	convertToDLL(root->right);
}// complexity O(n)

int heightOfBinaryTree(Node* root) {
	if (root == nullptr) return 0;
	int lh = heightOfBinaryTree(root->left);
	int rh = heightOfBinaryTree(root->right);
	return max(lh, rh) + 1;
}// complexity O(n)

int diameterOfBinaryTree(Node* root) {
	if(root==nullptr) return 0;
	int lh = heightOfTree(root->left);
	int rh = heightOfTree(root->right);
	int ld = diameterOfBinaryTree(root->left);
	int rd = diameterOfBinaryTree(root->right);
	return max(lh+rh+1, max(ld, rd));
}// complexity O(n^2)

int diameterOfBinaryTree2(Node* root, int* height) {
	if (root == nullptr) {
		*height = 0;
		return 0;
	}
	int lh = 0, rh = 0;
	int ld = diameterOfBinaryTree2(root->left, &lh);
	int rd = diameterOfBinaryTree2(root->right, &rh);
	*height = max(lh, rh) + 1;
	return max(lh+rh+1, max(ld, rd));
}// complexity O(n)

int lowestCommonAncestor(Node* root, int n1, int n2) {
	if (root == nullptr) return -1;
	if (root->data == n1 || root->data == n2) return root->data;
	int left = lowestCommonAncestor(root->left, n1, n2);
	int right = lowestCommonAncestor(root->right, n1, n2);
	if (left != -1 && right != -1) return root->data;
	if (left != -1) return left;
	else return right;
}// complexity O(n)
	
int timeToBurnTree(Node* root, int leaf, int& res) {
	if (root == nullptr) return 0;
	if (root->data == leaf) return 1;
	int l = timeToBurnTree(root->left, leaf, res);
	int r = timeToBurnTree(root->right, leaf, res);
	if (l != 0) {
		res = max(res, l + heightOfTree(root->right));
		return l + 1;
	}
	if (r != 0) {
		res = max(res, r + heightOfTree(root->left));
		return r + 1;
	}
	return 0;
}// complexity O(n)

int searchBST(Node* root, int key) {
	if(root==nullptr) return -1;
	if(root->data==key) return root->data;
	if(root->data>key) return searchBST(root->left, key);
	else return searchBST(root->right, key);
}// complexity O(h)

Node* insertBST(Node* root, int key) {
	if(root==nullptr) return new Node(key);
	if(root->data>key) root->left = insertBST(root->left, key);
	else root->right = insertBST(root->right, key);
	return root;
}// complexity O(h)

Node* insertBSTIterative(Node* root, int key) {
	Node* curr = root;
	Node* parent = nullptr;
	Node* newnode = new Node(key);
	while (curr != nullptr) {
		parent = curr;
		if (curr->data > key) {
			curr = curr->left;
		}
		else {
			curr = curr->right;
		}
	}
	if (parent == nullptr) parent = newnode;
	else if (parent->data < key) parent->right = newnode;
	else parent->left = newnode;
}

int minValue(Node* root) {
	int minval = 0;
	Node* curr = root;
	while (curr != nullptr && curr->left != nullptr) {
		minval = curr->left->data;
		curr = curr->left;
	}
	return minval;
}
Node* deleteBST(Node* root, int key) {
	if (root == nullptr) return root;
	if (key < root->data) root->left = deleteBST(root->right, key);
	else if (key > root->data) root->right = deleteBST(root->left, key);
	else {
		if (root->left == nullptr) return root->right;
		else if (root->right == nullptr) return root->left;
		root->data = minValue(root->right);
		root->right = deleteBST(root->right, root->data);
	}
}


int minBST(Node* root) {
	if (root == nullptr) return INT_MAX;
	else if (root->left == nullptr) return root->data;
	else return minBST(root->left);
}
int maxBST(Node* root) {
	if (root == nullptr) return INT_MIN;
	else if (root->right == nullptr) return root->data;
	else return maxBST(root->right);
}
int checkValidBST(Node* root) {
	if (root == nullptr) return 1; // Empty tree is a valid BST

	if ((root->left != nullptr && maxBST(root->left) >= root->data) ||
		(root->right != nullptr && minBST(root->right) <= root->data)) {
		return 0; // Not a BST
	}

	if (!checkValidBST(root->left) || !checkValidBST(root->right)) {
		return 0; // Not a BST
	}

	return 1; // Valid BST
}
int isBST1(Node* root) {
	if (root == nullptr) return 1; // Empty tree is a valid BST

	if (!checkValidBST(root)) {
		return 0; // Not a BST
	}

	if (!isBST1(root->left) || !isBST1(root->right)) {
		return 0; // Not a BST
	}

	return 1; // Valid BST
}

void inOrder2(Node* root, vector<int>& v) {
	if (root == nullptr) return;
	inOrder2(root->left, v);
	v.push_back(root->data);
	inOrder2(root->right, v);
}
int isBST2(Node* root) {
	vector<int> v;
	inOrder2(root, v);
	for (int i = 1; i < v.size(); i++) {
		if (v[i] <= v[i - 1]) return 0;
	}
	return 1;
}

int isBST3(Node* root, int min, int max) {
	if (root == nullptr) return 1;
	if (root->data < min || root->data > max) return 0;
	return isBST3(root->left, min, root->data - 1) & isBST3(root->right, root->data + 1, max);
}

int isBST4(Node* root, Node* l = nullptr, Node* r = nullptr) {
	if (root == nullptr) return 1;
	if (l != nullptr && root->data <= l->data) return 0;
	if (r != nullptr && root->data >= r->data) return 0;
	return isBST4(root->left, l, root) & isBST4(root->right, root, r);
}

int isBST5(Node* root) {
	static Node* prev = nullptr;
	if (root == nullptr) return 1;
	if (!isBST5(root->left)) return 0;
	if (prev != nullptr && root->data <= prev->data) return 0;
	prev = root;
	return isBST5(root->right);
}


int floorBST(Node* root, int key) {
	vector<int> v;
	inOrder2(root, v);
	int res = INT_MAX;
	for (int i = 0; i < v.size(); i++) {
		if (v[i] == key) return v[i];
		else if (v[i] < key) res = v[i];
		else break;
	}
	return res;
}
int ceilBST(Node* root, int key) {
	vector<int> v;
	inOrder2(root, v);
	int res = INT_MIN;
	for (int i = 0; i < v.size(); i++) {
		if (v[i] == key) return v[i];
		else if (v[i] > key) res = v[i];
		else break;
	}
	return res;
}

int floorBST2(Node* root, int key) {
	int res = INT_MAX;
	while (root != nullptr) {
		if (root->data == key) return root->data;
		else if (root->data > key) root = root->left;
		else {
			res = root->data;
			root = root->right;
		}
	}
	return res;
}
int ceilBST2(Node* root, int key) {
	int res = INT_MIN;
	while (root != nullptr) {
		if (root->data == key) return root->data;
		else if (root->data < key) root = root->right;
		else {
			res = root->data;
			root = root->left;
		}
	}
	return res;
}

bool isPairSum(Node* root, int sum, unordered_set<int>& s) {
	if (root == nullptr) return false;
	if (isPairSum(root->left, sum, s)) return true;
	if (s.find(sum - root->data) != s.end()) return true;
	else s.insert(root->data);
	return isPairSum(root->right, sum, s);
}  // complexity O(n)

vector<vector<int>> verticalOrderTraversal(Node* root) {
	vector<vector<int>> res;
	if (root == nullptr) return res;
	map<int, vector<int>> m;
	queue<pair<Node*, int>> q;
	q.push({ root, 0 });
	while (!q.empty()) {
		auto p = q.front();
		q.pop();
		Node* node = p.first;
		int hd = p.second;
		m[hd].push_back(node->data);
		if (node->left) q.push({ node->left, hd - 1 });
		if (node->right) q.push({ node->right, hd + 1 });
	}
	for (auto x : m) res.push_back(x.second);
	return res;
} // complexity O(nlogn)

vector<vector<int>> verticalTraversal2(Node* root) {
	vector<vector<int>> ans;
	queue<pair<Node*, pair<int, int>>> q;
	map<int, map<int, multiset<int>>> m;
	if (root == NULL)
		return ans;
	q.push({ root,{0,0} });
	while (!q.empty())
	{
		auto it = q.front();
		q.pop();
		Node* t = it.first;
		int v = it.second.first;
		int l = it.second.second;
		m[v][l].insert(t->data);
		if (t->left)
			q.push({ t->left,{v - 1,l + 1} });
		if (t->right)
			q.push({ t->right,{v + 1,l + 1} });
	}

	for (auto it : m)
	{
		vector<int> temp;
		for (auto i : it.second)
		{
			for (auto j : i.second)
			{
				temp.push_back(j);
			}
		}
		ans.push_back(temp);
	}
	return ans;
}

vector<vector<int>> verticalTraversal(Node* root) {
	map<int, map<int, multiset<int>>> mp; // multiset to get the sorted order
	queue<pair<Node*, pair<int, int>>> q; // [node,line,level]
	q.push({ root,{0,0} });
	while (!q.empty()) {
		auto it = q.front();
		q.pop();
		Node* node = it.first;
		int line = it.second.first;
		int level = it.second.second;
		mp[line][level].insert(node->data);
		if (node->left) q.push({ node->left,{line - 1,level + 1} });
		if (node->right) q.push({ node->right,{line + 1,level + 1} });
	}
	vector<vector<int>> ans;
	for (auto itr1 : mp) {
		vector<int> temp;
		for (auto itr2 : itr1.second) {
			temp.insert(temp.end(), itr2.second.begin(), itr2.second.end());
		}
		ans.push_back(temp);
	}
	return ans;
}

void printVerticalOrder(Node* root) {
	map<int, vector<int>> m;
	queue<pair<Node*, int>> q;
	q.push({ root, 0 });
	while (!q.empty()) {
		auto p = q.front();
		q.pop();
		Node* node = p.first;
		int hd = p.second;
		m[hd].push_back(node->data);
		if (node->left) q.push({ node->left, hd - 1 });
		if (node->right) q.push({ node->right, hd + 1 });
	}
	for (auto x : m) {
		for (auto y : x.second) {
			cout << y << " ";
		}
		cout << endl;
	}
} // complexity O(nlogn)

class AVLTree {
	Node* root;
	int height(Node* root) {
		if (root == nullptr) return 0;
		return root->height;
	}
	int getBalance(Node* root) {
		if (root == nullptr) return 0;
		return height(root->left) - height(root->right);
	}
	Node* rightRotate(Node* root) {
		Node* newroot = root->left;
		Node* temp = newroot->right;
		newroot->right = root;
		root->left = temp;
		root->height = 1 + max(height(root->left), height(root->right));
		newroot->height = 1 + max(height(newroot->left), height(newroot->right));
		return newroot;
	}
	Node* leftRotate(Node* root) {
		Node* newroot = root->right;
		Node* temp = newroot->left;
		newroot->left = root;
		root->right = temp;
		root->height = 1 + max(height(root->left), height(root->right));
		newroot->height = 1 + max(height(newroot->left), height(newroot->right));
		return newroot;
	}
	Node* insert(Node* root, int key) {
		if (root == nullptr) return new Node(key);
		if (key < root->data) root->left = insert(root->left, key);
		else if (key > root->data) root->right = insert(root->right, key);
		else return root;
		root->height = 1 + max(height(root->left), height(root->right));
		int balance = getBalance(root);
		if (balance > 1 && key < root->left->data) return rightRotate(root);
		if (balance < -1 && key > root->right->data) return leftRotate(root);
		if (balance > 1 && key > root->left->data) {
			root->left = leftRotate(root->left);
			return rightRotate(root);
		}
		if (balance < -1 && key < root->right->data) {
			root->right = rightRotate(root->right);
			return leftRotate(root);
		}
		return root;
	}
	Node* deleteNode(Node* root, int key) {
		if (root == nullptr) return root;
		if (key < root->data) root->left = deleteNode(root->left, key);
		else if (key > root->data) root->right = deleteNode(root->right, key);
		else {
			if (root->left == nullptr || root->right == nullptr) {
				Node* temp = root->left ? root->left : root->right;
				if (temp == nullptr) {
					temp = root;
					root = nullptr;
				}
				else *root = *temp;
				delete temp;
			}
			else {
				Node* temp = root->right;
				while (temp->left != nullptr) temp = temp->left;
				root->data = temp->data;
				root->right = deleteNode(root->right, temp->data);
			}
		}
		if (root == nullptr) return root;
		root->height = 1 + max(height(root->left), height(root->right));
		int balance = getBalance(root);
		if (balance > 1 && getBalance(root->left) >= 0) return rightRotate(root);
		if (balance > 1 && getBalance(root->left) < 0) {
			root->left = leftRotate(root->left);
			return rightRotate(root);
		}
		if (balance < -1 && getBalance(root->right) <= 0) return leftRotate(root);
		if (balance < -1 && getBalance(root->right) > 0) {
			root->right = rightRotate(root->right);
			return leftRotate(root);
		}
		return root;
	}	
};
class adjMatrixGraph {
	int v;
	int e;
	vector<vector<int>> adj;
	public:
		adjMatrixGraph(int v, int e) {
		this->v = v;
		this->e = e;
		adj.resize(v, vector<int>(v, 0));
		}
		void addEdge(int u, int v) {
		adj[u][v] = 1;
		adj[v][u] = 1;
		}
		void printGraph() {
			for (int i = 0; i < v; i++) {
			cout << i << " -> ";
			for (int j = 0; j < v; j++) {
				if (adj[i][j] == 1) cout << j << " ";
			}
			cout << endl;
			}
		}
};
class adjListGraph {
	int v;
	int e;
	vector<vector<int>> adj;
	public:
		adjListGraph(int v, int e) {
		this->v = v;
		this->e = e;
		adj.resize(v);
		}
		void addEdge(int u, int v) {
		adj[u].push_back(v);
		adj[v].push_back(u);
		}
		void printGraph() {
			for (int i = 0; i < v; i++) {
			cout << i << " -> ";
			for (int j = 0; j < adj[i].size(); j++) {
				cout << adj[i][j] << " ";
			}
			cout << endl;
			}
		}
};

bool graphBFS(vector<vector<int>> adj, int src, int des, int v, vector<int> pred, vector<int> dist) {
	queue<int> q;
	vector<bool> visited(v, false);
	for (int i = 0; i < v; i++) {
		dist[i] = INT_MAX;
		pred[i] = -1;
	}
	visited[src] = true;
	dist[src] = 0;
	q.push(src);
	while (!q.empty()) {
		int u = q.front();
		q.pop();
		for (int i = 0; i < adj[u].size(); i++) {
			if (visited[adj[u][i]] == false) {
				visited[adj[u][i]] = true;
				dist[adj[u][i]] = dist[u] + 1;
				pred[adj[u][i]] = u;
				q.push(adj[u][i]);
				if (adj[u][i] == des) return true;
			}
		}
	}
	return false;
}

int connectedComponents(vector<vector<int>> adj, int v) {
	vector<bool> visited(v, false);
	int count = 0;
	for (int i = 0; i < v; i++) {
		if (visited[i] == false) {
			count++;
			queue<int> q;
			q.push(i);
			visited[i] = true;
			while (!q.empty()) {
				int u = q.front();
				q.pop();
				for (int j = 0; j < adj[u].size(); j++) {
					if (visited[adj[u][j]] == false) {
						visited[adj[u][j]] = true;
						q.push(adj[u][j]);
					}
				}
			}
		}
	}
	return count;
}	

vector<int>  dfsOfGraph(vector<vector<int>> adj, int v) {
	vector<bool> visited(v, false);
	vector<int> res;
	for (int i = 0; i < v; i++) {
		if (visited[i] == false) {
			stack<int> st;
			st.push(i);
			visited[i] = true;
			while (!st.empty()) {
				int u = st.top();
				st.pop();
				res.push_back(u);
				for (int j = 0; j < adj[u].size(); j++) {
					if (visited[adj[u][j]] == false) {
						visited[adj[u][j]] = true;
						st.push(adj[u][j]);
					}
				}
			}
		}
	}
	return res;
}
void dfsOfGraph2Util(vector<vector<int>> adj, int u, vector<bool>& visited, vector<int>& res) {
	visited[u] = true;
	res.push_back(u);
	for (int i = 0; i < adj[u].size(); i++) {
		if (visited[adj[u][i]] == false) {
			dfsOfGraph2Util(adj, adj[u][i], visited, res);
		}
	}
}
vector<int> dfsOfGraph2(vector<vector<int>> adj, int v) {
	vector<bool> visited(v, false);
	vector<int> res;
	for (int i = 0; i < v; i++) { // for disconnected graph
		if (visited[i] == false) {
			dfsOfGraph2Util(adj, i, visited, res);
		}
	}
	return res;
}

bool cycleInUndirectedGraphUtil(int u, vector<bool>& visited, vector<vector<int>> adj, int parent) {
	visited[u] = true;
	for (int i = 0; i < adj[u].size(); i++) {
		if (visited[adj[u][i]] == false) {
			if (cycleInUndirectedGraphUtil(adj[u][i], visited, adj, u)) return true;
		}
		else if (adj[u][i] != parent) return true;
	}
	return false;
}
bool cycleInUndirectedGraph(int v, vector<vector<int>> adj) {
	vector<bool> visited(v, false);
	for (int i = 0; i < v; i++) {
		if (visited[i] == false) {
			if(cycleInUndirectedGraphUtil(i, visited, adj, -1)) return true;
		}
	}
	return false;
}



int main() {
	
	return 0;
}

