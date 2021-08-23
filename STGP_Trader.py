""" GP_Entity will initialise multiple of these and give them each an improvement function dictated by the tree of the individual."""

from typing import List
import random

from BSE2_msg_classes import Order
from BSE2_trader_agents import Trader
import BSE2_sys_consts

class Order_Data():
    """ used for logging """
    def __init__(self, customer_price, trans_price, posted_improve, actual_improve, exchange_msg):
        self.customer_price = customer_price
        self.trans_price = trans_price
        self.posted_improve = posted_improve
        self.actual_improve = actual_improve
        self.exchange_msg = exchange_msg

    def __repr__(self):
        return str(self.__dict__)


class Generation_Data():
    
    def __init__(self, tid: str, gen_num: int):
        self.tid = tid
        self.gen_num = gen_num
        self.transactions: List[Order_Data] = []

    def __repr__(self):
        return str(self.__dict__)


class STGP_Trader(Trader):
    
    def __init__(self, tid, balance, time, trading_func):
        super().__init__("STGP", tid, balance, time)

        # trading function from STGP tree (calculates improvement on customer order).
        self.trading_func = trading_func

        # profit tracking
        self.last_evolution = 0.0
        self.current_gen = 0
        self.profit_since_evolution = 0.0
        self.generational_profits = []

        # Exponential Moving Average
        self.ema = None
        self.nLastTrades = 5
        self.ema_param = 2 / float(self.nLastTrades + 1)

        # transaction price tracking
        self.last_t_price = None

        # Stat tracking
        self.all_gens_data = []
        self.current_gen_data = Generation_Data(tid, self.current_gen)
        self.orderprices = []
        self.quoteprices = []
    
    def del_cust_order(self, cust_order_id, verbose):
        super().del_cust_order(cust_order_id, verbose)

    def get_profit(self, time):
        """ Gets the profit of the current generation. Used for gp evaluation. """
        if time == 0: return 0
        return self.profit_since_evolution

    def reset_gen_profits(self):
        """ Called after the expr has been updated due to evolution. This is necessary to evaluate a generation. """
        self.generational_profits.append(self.profit_since_evolution)
        self.profit_since_evolution = 0
        self.current_gen += 1

        # logging
        self.all_gens_data.append(self.current_gen_data)
        self.current_gen_data = Generation_Data(self.tid, self.current_gen)


    def _update_ema(self, price):
        """ Update exponential moving average indicator for the trader. """
        if self.ema == None: self.ema = price
        else: self.ema = self.ema_param * price + (1 - self.ema_param) * self.ema

    def getorder(self, time, countdown, lob, verbose):
        """ Called by the market session to get an order from this trader. """
        if len(self.orders) < 1:
            # no orders: return NULL
            order = None
        else:
            self.limit = self.orders[0].price
            self.job = self.orders[0].atype


            improvement = 0


            # ######## CONNECT PRIMITIVE SET VARIABLES ###########
            # improvement is always positive and related to customer limit price ...
            # ... ensuring symmetrical behaviour between BUY and SELL.


            # stub price : maximum possible improvement
            if self.job == "Bid":
                stub = BSE2_sys_consts.bse_sys_maxprice - self.limit
            elif self.job == "Ask":
                stub = self.limit - BSE2_sys_consts.bse_sys_minprice
            else:
                raise ValueError("Invalid job type")


            # last transaction price : difference between ltp and customer limit price
            # no boolean 'is present' check as absense is rare and useless primitive will hinder evolution
            if self.last_t_price == None:
                ltp = BSE2_sys_consts.bse_sys_maxprice
            else:
                if self.job == "Bid":
                    ltp = self.limit - self.last_t_price
                elif self.job == "Ask":
                    ltp = self.last_t_price - self.limit


            # same present : is an order present on the same side of the order book as the trader?
            # opposite present : is an order present on the opposite side of the order book as the trader?
            same_present = True
            opp_present = True
            if self.job == 'Bid':
                if lob['bids']['bestp'] == None:
                    same_present = False
                if lob['asks']['bestp'] == None:
                    opp_present = False
            elif self.job == 'Ask':
                if lob['asks']['bestp'] == None:
                    same_present = False
                if lob['bids']['bestp'] == None:
                    opp_present = False


            # best same: best price on the same side of the order book as the trader
            # best opp: best price on the other side of the order book
            if self.job == 'Bid':
                if lob['bids']['bestp'] == None: 
                    # no bids present
                    best_same = stub
                else:
                    best_same = self.limit - lob['bids']['bestp']
                
                if lob['asks']['bestp'] == None:
                    # stub chosen for both same and opp as unknown how the number will be used. 
                    best_opp = stub
                else:
                    best_opp = self.limit - lob['asks']['bestp']

            elif self.job == 'Ask':
                if lob['asks']['bestp'] == None:
                    best_same = stub
                else:
                    best_same = lob['asks']['bestp'] - self.limit
                
                # opposite side
                if lob['bids']['bestp'] == None:
                    best_opp = stub
                else:
                    best_opp = lob['bids']['bestp'] - self.limit

            else:
                raise ValueError('Invalid job type')


            # worst same: worst price on the same side of the order book as the trader
            # worst opp: worst price on the other side of the order book
            if self.job == 'Bid':
                if lob['bids']['worstp'] == None: 
                    # no bids present
                    worst_same = stub
                else:
                    worst_same = self.limit - lob['bids']['worstp']
                
                if lob['asks']['worstp'] == None:
                    # stub chosen for both same and opp as unknown how the number will be used. 
                    worst_opp = stub
                else:
                    worst_opp = self.limit - lob['asks']['worstp']

            elif self.job == 'Ask':
                if lob['asks']['worstp'] == None:
                    worst_same = stub
                else:
                    worst_same = lob['asks']['worstp'] - self.limit
                
                # opposite side
                if lob['bids']['worstp'] == None:
                    worst_opp = stub
                else:
                    worst_opp = lob['bids']['worstp'] - self.limit

            else:
                raise ValueError('Invalid job type')


            # random : a random improvement between 0 and max improvement.
            rand = random.randint(0, stub)

            
            # #############################################################


            # calculate improvement on customer order via STGP function
            improvement = self.trading_func(stub, ltp, same_present, opp_present, best_same, best_opp,
                                 worst_same, worst_opp, rand, countdown, time)

            # reset negative improvements to 0
            if improvement < 0:
                improvement = 0



            if verbose:
                print(f"trader: {self.tid}, limit price: {self.limit}, improvement found: {improvement}")



            # print(f"trader: {self.tid}, expr:{self.trading_func}, limit price: {self.limit}, improvement found: {improvement}")


            # improvement is added to customer asks and subtracted to customer bids - achieving symmetrical behaviour
            if self.job == 'Bid':
                quoteprice = int(self.limit - improvement)
                if quoteprice < BSE2_sys_consts.bse_sys_minprice:
                    quoteprice = BSE2_sys_consts.bse_sys_minprice
                if quoteprice > self.limit:
                    quoteprice = self.limit

            elif self.job == 'Ask':
                quoteprice = int(self.limit + improvement)
                if quoteprice > BSE2_sys_consts.bse_sys_maxprice:
                    quoteprice = BSE2_sys_consts.bse_sys_maxprice
                if quoteprice < self.limit:
                    quoteprice = self.limit
            


            order = Order(self.tid, self.job, "LIM", quoteprice, 
                          self.orders[0].qty, time, None, -1)

            self.orderprices.append(self.limit)
            self.quoteprices.append((time, quoteprice, self.limit))

            self.price = quoteprice
            self.lastquote = order

        return order

    def respond(self, time, lob, trade, verbose):
        """ Called by the market session to notify trader of LOB updates. """
        if (trade != None):
            self._update_ema(trade["price"]) # update EMA
            self.last_t_price = trade["price"]


    def bookkeep(self, msg, time, verbose):
        if msg.event == "FILL":
            improvement = abs(msg.trns[0]["Price"] - self.orders[0].price)
            self.profit_since_evolution += improvement

            # for logging: customer price, trans price, attemted improvement, actual improvement, msg
            posted_improvement = abs(self.lastquote.price - self.limit)
            new_order_data = Order_Data(self.limit, msg.trns[0]["Price"], 
                                        posted_improvement, improvement, msg)
            self.current_gen_data.transactions.append(new_order_data)
        
        super().bookkeep(msg, time, verbose)

    def get_gen_profits(self):
        """ to be called at the end of experiment """
        ps = self.generational_profits
        ps.append(self.profit_since_evolution)
        return ps
