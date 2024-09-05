import { Outlet } from "react-router-dom";
import FooterLinks from "../Footer/Footer";
import HeaderMegaMenu from "../Header/Header";
export default function MainLayout(){
    return (
       <div>
        <HeaderMegaMenu/>
            <Outlet/>
        <FooterLinks/>
        </div>
    )
}