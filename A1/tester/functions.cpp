#include "functions.h"

using namespace std;

class OrderBookEntry {
    public:
        uint32_t stockID;
        bool orderType;     // false=buy, true=sell
        uint8_t orderQty;
        uint8_t orderValue;

    OrderBookEntry() : stockID(0), orderType(false), orderQty(0), orderValue(0) {}
};

struct StockInfo {
    uint8_t lastBuyValue;
    uint8_t lastSellValue;
    bool hasBuy;
    bool hasSell;

    StockInfo() : lastBuyValue(0), lastSellValue(0), hasBuy(false), hasSell(false) {}

    int getSpread() const {
        return abs((int)lastSellValue - (int)lastBuyValue);
    }
};

// Helper Functions
/*
 Prompt Used : write a function to remove bit stuffing as per the rule = ""
The packet is constructed in a binary format, starting from the LSB:
1. The first 32 bits indicate the stockID.
2. The next bit indicates a buy (0) or a sell(1) order.
3. The next 8 bits indicate the orderQty.
4. The last 8 bits indicate the orderValue.
Now this 49 bit packet is padded with zeroes to prevent any occurrence of 6 consecutive 1’s
(6 consecutive 1’s represent the packet boundary). This is done by inserting a 0 after every 5
consecutive 1’s

*/
uint64_t removeBitStuffing(uint64_t encoded) {
    uint64_t decoded = 0;
    int decodedPos = 0;
    int consecutiveOnes = 0;
    const int TARGET_BITS = 49;
    
    int i = 0;
    while (i < 64 && decodedPos < TARGET_BITS) {
        int bit = (encoded >> i) & 0x1;
        
        if (bit == 1) {
            consecutiveOnes++;
            decoded |= (1ULL << decodedPos);
            decodedPos++;
            
            if (consecutiveOnes == 5) {
                consecutiveOnes = 0;
                i += 2;
                continue;
            }
        } else {
            consecutiveOnes = 0;
            decodedPos++;
        }
        
        i++;
    }
    
    return decoded;
}

OrderBookEntry decodePacket(uint64_t encodedPacket) {
    uint64_t decoded = removeBitStuffing(encodedPacket);
    OrderBookEntry entry;

    entry.stockID = decoded & 0xFFFFFFFF;
    decoded >>= 32;

    entry.orderType = decoded & 1;
    decoded >>= 1;

    entry.orderQty = decoded & 0xFF;
    decoded >>= 8;
    
    entry.orderValue = decoded & 0xFF;
    return entry;
}



void generateSnapShot(int snapShotID, const map<uint32_t, StockInfo>& stockData) {
    string filename = "snap_" + to_string(snapShotID) + ".txt";
    ofstream outFile(filename);

    if(!outFile.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return;
    }

    struct SnapShotEntry {
        uint32_t stockID;
        uint8_t lastSellValue;
        uint8_t lastBuyValue;
        int spread;
    };
    vector<SnapShotEntry> snapShotEntries;
    for(const auto& [stockID, stockInfo]:stockData) {
        SnapShotEntry entry;
        entry.stockID = stockID;
        entry.lastBuyValue = stockInfo.lastBuyValue;
        entry.lastSellValue = stockInfo.lastSellValue;
        entry.spread = stockInfo.getSpread();
        snapShotEntries.push_back(entry);
    }

    sort(snapShotEntries.begin(), snapShotEntries.end(), [](const auto& a, const auto& b) {
        if(a.spread != b.spread) return a.spread > b.spread;
        else return a.stockID > b.stockID;
    });

    for(auto& entry:snapShotEntries)
        outFile << entry.stockID << " " << (int)entry.lastSellValue << " " << (int)entry.lastBuyValue << " " << entry.spread << "\n";
    
    outFile.close();
}

void updateDisplay(const std::vector<uint64_t> &orderBook, int32_t freq){
    int n = orderBook.size();
    int numSanpShots = 1 + (n/freq);

    vector<map<uint32_t, StockInfo>> snapShotData(numSanpShots);
    map<uint32_t, StockInfo> currentData;
    int currentSnapShotID = 0;

    for(int i=0;i<n;i++) {
        OrderBookEntry entry = decodePacket(orderBook[i]);
        if(currentData.find(entry.stockID) == currentData.end())
            currentData[entry.stockID] = StockInfo();
        if(entry.orderType == 0) {
            currentData[entry.stockID].lastBuyValue = entry.orderValue;
            currentData[entry.stockID].hasBuy = true;
        }
        else {
            currentData[entry.stockID].lastSellValue = entry.orderValue;
            currentData[entry.stockID].hasSell = true;
        }

        if((i+1)%freq == 0) {
            snapShotData[currentSnapShotID] = currentData;
            currentSnapShotID++;

            if(n%freq == 0 && (i+1)==n) {
                snapShotData[currentSnapShotID] = currentData;
                currentSnapShotID++;
            }
        }
    }

    #pragma omp parallel for schedule(dynamic)
    for(int i=0;i<numSanpShots;i++)
        generateSnapShot(i, snapShotData[i]);
}

int64_t totalAmountTraded(const std::vector<uint64_t> &orderBook)
{
    int n = orderBook.size();
    int64_t totalAmout = 0;

    #pragma omp parallel for reduction(+:totalAmout)
    for(int i=0;i<n;i++) {
        OrderBookEntry entry = decodePacket(orderBook[i]);
        int64_t tradeAmount = (int64_t)entry.orderQty * (int64_t)entry.orderValue;
        totalAmout += tradeAmount;
    }

    return totalAmout;
}

void printOrderStats(const std::vector<uint64_t> &orderBook)
{
    int n = orderBook.size();
    
    struct StockStats {
        uint32_t stockID;
        uint8_t minSellValue;
        uint8_t maxBuyValue;
        int64_t totalValue;
        int64_t orderCount;
        bool hasSell;
        bool hasBuy;

        StockStats() : stockID(0), minSellValue(255), maxBuyValue(0), totalValue(0), orderCount(0), hasSell(false), hasBuy(false) {}
    };

    map<uint32_t, StockStats> statsData;

    for(int i=0;i<n;i++) {
        OrderBookEntry entry = decodePacket(orderBook[i]);
        if(statsData.find(entry.stockID) == statsData.end()) {
            statsData[entry.stockID] = StockStats();
            statsData[entry.stockID].stockID = entry.stockID;
        }

        //sell
        if(entry.orderType) {
            statsData[entry.stockID].hasSell = true;
            statsData[entry.stockID].minSellValue = min(entry.orderValue, statsData[entry.stockID].minSellValue);
        }
        else { // Buy
            statsData[entry.stockID].hasBuy = true;
            statsData[entry.stockID].maxBuyValue = max(entry.orderValue, statsData[entry.stockID].maxBuyValue);
        }

        statsData[entry.stockID].totalValue += entry.orderValue;
        statsData[entry.stockID].orderCount++;
    }

    vector<StockStats> statsList;
    for(auto& [stockID, stat]:statsData)
        statsList.push_back(stat);

    sort(statsList.begin(), statsList.end(), [](const auto& a, const auto& b) {
        return a.stockID < b.stockID;
    });

    ofstream outFile("stats.txt");
    if(!outFile.is_open()) {
        cerr << "Error opening stats.txt" << endl;
        return;
    }

    outFile << fixed << setprecision(4);
    for(auto& entry:statsList) {
        double avgValue = (double)entry.totalValue / (double)entry.orderCount;
        uint8_t minSell = entry.hasSell ? entry.minSellValue : 0;
        uint8_t maxBuy = entry.hasBuy ? entry.maxBuyValue : 0;

        outFile << entry.stockID << " " << (int)minSell << " " << (int)maxBuy << " " << avgValue << "\n";
    }
    outFile.close();
}